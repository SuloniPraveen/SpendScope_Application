import os
import json
from pathlib import Path

import pdfplumber
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # for server environments (no GUI)
import matplotlib.pyplot as plt

from dateutil import parser as date_parser
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# 0. Setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Load .env if present
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    print("❌ OPENAI_API_KEY not set. Set it in .env or export it in your shell.")
    client = None
else:
    print("✅ Loaded OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_KEY)

app = Flask(__name__)
app.secret_key = "super-secret-key"  # change in prod

CURRENCY_SYMBOL = "$"  # used in templates


# =========================================================
# 1. PDF → raw text
# =========================================================

def extract_text_from_pdf(file_storage) -> str:
    file_storage.stream.seek(0)
    with pdfplumber.open(file_storage.stream) as pdf:
        pages_text = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    full_text = "\n".join(pages_text)
    print("\n----- PDF TEXT SNIPPET -----")
    print(full_text[:1000])
    print("----- END SNIPPET -----\n")
    return full_text


def extract_transaction_section(raw_text: str) -> str:
    """
    Try to keep only the transaction table part (for speed + accuracy).

    Heuristic:
    - Start after a line containing "Transaction Date" and "Value Date"
    - If not found, return the whole text.
    """
    lines = raw_text.splitlines()
    keep = False
    kept_lines = []

    for line in lines:
        if ("Transaction Date" in line and "Value Date" in line) or ("Tran Date" in line and "Value Date" in line):
            keep = True
            continue
        if keep:
            kept_lines.append(line)

    if kept_lines:
        section = "\n".join(kept_lines)
        print("Using only transaction section (trimmed text).")
        return section

    print("No obvious header found, using full text for AI.")
    return raw_text


def salvage_transactions_from_truncated_json(raw: str):
    """
    If the AI JSON is truncated (e.g. cut in the middle of a transaction),
    salvage all *complete* transaction objects from the 'transactions' array.

    Returns a Python dict like {"transactions": [...] } or None if it fails.
    """
    try:
        # First try normal parse
        return json.loads(raw)
    except Exception:
        pass  # we'll try to salvage below

    key_idx = raw.find('"transactions"')
    if key_idx == -1:
        return None

    bracket_idx = raw.find('[', key_idx)
    if bracket_idx == -1:
        return None

    i = bracket_idx + 1
    depth = 0
    last_good_end = None

    # Scan for complete {...} objects inside the array
    while i < len(raw):
        c = raw[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                # End of a full transaction object
                last_good_end = i
        i += 1

    if last_good_end is None:
        return None

    body = raw[bracket_idx + 1: last_good_end + 1]
    body = body.rstrip(", \n\r\t")

    safe_json = '{"transactions": [' + body + ']}'

    try:
        return json.loads(safe_json)
    except Exception as e:
        print("Salvage JSON still failed:", repr(e))
        return None


# =========================================================
# 2. AI: text → transactions (JSON enforced, with category)
# =========================================================

def extract_transactions_with_ai(raw_text: str):
    """
    Use OpenAI to extract structured transactions from arbitrary bank statements.
    JSON mode is enforced, and if the output is truncated we salvage all
    complete transactions from the 'transactions' array.
    Each transaction also includes a semantic 'category' chosen by the model.
    """
    if client is None:
        print("No OpenAI client; cannot extract transactions.")
        return []

    if not raw_text.strip():
        return []

    text_snippet = raw_text[:20000]  # limit text so it runs fast

    system_msg = (
        "You are a financial data extractor. "
        "You ALWAYS return ONLY valid JSON (no commentary, no trailing text). "
        "You also classify each transaction into a spending category."
    )

    categories_list = [
        "Food & Dining",
        "Groceries",
        "Transport",
        "Shopping",
        "Rent & Utilities",
        "Entertainment",
        "Salary / Income",
        "Bills & Recharges",
        "Transfers & Payments",
        "Loans / EMIs",
        "Fees & Charges",
        "Cash Withdrawal",
        "Savings / Investments",
        "Other"
    ]

    user_prompt = f"""
From the bank statement text below, extract all individual transactions.

For EACH transaction, return:
- date: the transaction date in YYYY-MM-DD format
- description: a short useful description
- amount: negative for money going out (debits), positive for money coming in (credits)
- type: "debit" or "credit"
- category: the BEST MATCH from this list:
  {categories_list}

Return STRICT JSON in this format (no extra fields, no comments, no text outside JSON):

{{
  "transactions": [
    {{
      "date": "YYYY-MM-DD",
      "description": "some text",
      "amount": -123.45,
      "type": "debit",
      "category": "Food & Dining"
    }}
  ]
}}

Bank statement text:
\"\"\"{text_snippet}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=6000,
        )
    except Exception as e:
        print("AI extraction error:", repr(e))
        return []

    content = response.choices[0].message.content

    # 1) Normal parse
    try:
        data = json.loads(content)
        txns = data.get("transactions", [])
        print(f"AI extracted {len(txns)} transactions (direct JSON).")
        return txns
    except Exception as e:
        print("JSON parse error:", repr(e))
        print("JSON parse error:", e)

    # 2) Salvage partial JSON if truncated
    salvaged = salvage_transactions_from_truncated_json(content)
    if salvaged and "transactions" in salvaged:
        txns = salvaged["transactions"]
        print(f"AI extracted {len(txns)} transactions (salvaged).")
        return txns

    print("Failed to extract any transactions from AI output.")
    return []


# =========================================================
# 3. Transactions → DataFrame (use AI category)
# =========================================================

def transactions_to_df(txns):
    if not txns:
        return pd.DataFrame(columns=["date", "description", "amount", "type", "abs_amount", "category"])

    df = pd.DataFrame(txns)

    # Normalize / clean
    df["description"] = df["description"].fillna("").astype(str)

    def parse_date_safe(d):
        try:
            return date_parser.parse(d).date()
        except Exception:
            return None

    df["date"] = df["date"].apply(parse_date_safe)
    df = df.dropna(subset=["date"])

    df["amount"] = df["amount"].astype(float)

    def infer_type(row):
        t = str(row.get("type", "")).lower()
        if t in ("debit", "credit"):
            return t
        return "debit" if row["amount"] < 0 else "credit"

    df["type"] = df.apply(infer_type, axis=1)
    df["abs_amount"] = df["amount"].abs()

    # Use AI-provided category if available
    if "category" in df.columns:
        df["category"] = df["category"].fillna("Other").astype(str)
    else:
        df["category"] = "Other"

    print(f"DataFrame has {len(df)} rows after cleaning.")
    return df


# =========================================================
# 4. Analytics + charts
# =========================================================

def compute_insights(df: pd.DataFrame):
    """
    Returns:
      summary, monthly_spend_dict, category_spend_dict,
      top_categories, category_chart_path, monthly_chart_path, other_breakdown
    """
    if df.empty:
        return {}, {}, {}, [], None, None, []

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    monthly_spend = (
        df[df["type"] == "debit"]["abs_amount"]
        .resample("M")
        .sum()
        .rename("monthly_spend")
    )

    category_spend = (
        df[df["type"] == "debit"]
        .groupby("category")["abs_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    total_debit = df[df["type"] == "debit"]["abs_amount"].sum()
    total_credit = df[df["type"] == "credit"]["abs_amount"].sum()
    days_covered = (df.index.max() - df.index.min()).days + 1
    avg_daily_spend = total_debit / days_covered if days_covered > 0 else 0

    summary = {
        "total_debit": round(total_debit, 2),
        "total_credit": round(total_credit, 2),
        "days_covered": days_covered,
        "avg_daily_spend": round(avg_daily_spend, 2),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
    }

    monthly_spend_dict = {d.strftime("%Y-%m"): round(v, 2) for d, v in monthly_spend.items()}
    category_spend_dict = {k: round(v, 2) for k, v in category_spend.items()}
    top_categories = list(category_spend_dict.items())

    # Breakdown of what "Other" actually contains (top descriptions)
    other_breakdown = []
    if "Other" in category_spend.index:
        other_df = df[df["category"] == "Other"]
        if not other_df.empty:
            grouped = (
                other_df.groupby("description")["abs_amount"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            other_breakdown = [
                {"description": desc, "amount": round(val, 2)}
                for desc, val in grouped.items()
            ]

    category_chart = generate_category_chart(category_spend_dict)
    monthly_chart = generate_monthly_chart(monthly_spend_dict)

    return summary, monthly_spend_dict, category_spend_dict, top_categories, category_chart, monthly_chart, other_breakdown


def _ensure_plots_dir():
    plots_dir = BASE_DIR / "static" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def generate_category_chart(category_spend_dict):
    if not category_spend_dict:
        return None

    plots_dir = _ensure_plots_dir()

    categories = list(category_spend_dict.keys())
    amounts = list(category_spend_dict.values())

    plt.close("all")
    # Bigger figure
    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    fig.patch.set_facecolor("#050816")
    ax.set_facecolor("#020617")

    bars = ax.bar(categories, amounts, width=0.6, color="#4f46e5")

    ax.set_ylabel("Total Spend", color="#e5e7eb")
    ax.set_title("Spending by Category", color="#e5e7eb", pad=12)
    ax.tick_params(axis="x", labelrotation=35, labelsize=9, colors="#9ca3af")
    ax.tick_params(axis="y", labelsize=9, colors="#9ca3af")

    for spine in ax.spines.values():
        spine.set_color("#475569")

    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#334155")

    # Value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#e5e7eb",
        )

    fig.tight_layout()
    path = plots_dir / "category_spend.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return "/static/plots/category_spend.png"


def generate_monthly_chart(monthly_spend_dict):
    if not monthly_spend_dict:
        return None

    plots_dir = _ensure_plots_dir()

    months = list(monthly_spend_dict.keys())
    amounts = list(monthly_spend_dict.values())

    plt.close("all")
    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    fig.patch.set_facecolor("#050816")
    ax.set_facecolor("#020617")

    bars = ax.bar(months, amounts, width=0.6, color="#22c55e")

    ax.set_ylabel("Total Spend", color="#e5e7eb")
    ax.set_title("Monthly Spending", color="#e5e7eb", pad=12)
    ax.tick_params(axis="x", labelrotation=35, labelsize=9, colors="#9ca3af")
    ax.tick_params(axis="y", labelsize=9, colors="#9ca3af")

    for spine in ax.spines.values():
        spine.set_color("#475569")

    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#334155")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#e5e7eb",
        )

    fig.tight_layout()
    path = plots_dir / "monthly_spend.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return "/static/plots/monthly_spend.png"


# =========================================================
# 5. AI Recommendations (structured, expandable)
# =========================================================

def generate_recommendations_ai(summary, category_spend_dict, df):
    """
    Returns a list of recommendation dicts:
    {
      "title": "...",
      "detail": "...",
      "focus": "savings" | "debt" | "subscriptions" | "spending" | "other"
    }
    """
    if client is None:
        return [{
            "title": "AI unavailable",
            "detail": "AI recommendations are unavailable — OPENAI_API_KEY is not set.",
            "focus": "other",
        }]

    if df.empty:
        return []

    category_sorted = sorted(category_spend_dict.items(), key=lambda x: x[1], reverse=True)
    top_cats = category_sorted[:8]

    df_debit = df[df["type"] == "debit"].sort_values("abs_amount", ascending=False)
    df_sample = df_debit.head(12)[["date", "description", "abs_amount", "category"]]

    sample_txn = [
        {
            "date": str(row["date"]),
            "description": str(row["description"])[:80],
            "amount": float(row["abs_amount"]),
            "category": str(row["category"]),
        }
        for _, row in df_sample.iterrows()
    ]

    payload = {
        "summary": summary,
        "top_categories": top_cats,
        "sample_transactions": sample_txn,
    }

    prompt = f"""
You are a friendly personal finance coach.

The user's bank statement analytics are:

{json.dumps(payload, indent=2)}

Using this information, produce 4–6 structured recommendations.

Each recommendation should have:
- "title": a short, actionable headline (max 80 characters)
- "detail": 2–3 sentences explaining what to do and why, in simple, non-judgmental language
- "focus": one of ["savings", "debt", "subscriptions", "spending", "other"]

Return ONLY valid JSON in this format:

{{
  "recommendations": [
    {{
      "title": "Review food delivery spending",
      "detail": "Explain the recommendation details here...",
      "focus": "spending"
    }}
  ]
}}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You help people understand their spending and give practical, kind money tips.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=700,
        )
    except Exception as e:
        print("AI recommendation error:", repr(e))
        return [{
            "title": "AI recommendations unavailable",
            "detail": "There was an error calling the AI API. Try again in a little while.",
            "focus": "other",
        }]

    try:
        data = json.loads(resp.choices[0].message.content)
        recs = data.get("recommendations", [])
        cleaned = []
        for rec in recs:
            cleaned.append({
                "title": str(rec.get("title", "")).strip() or "Recommendation",
                "detail": str(rec.get("detail", "")).strip(),
                "focus": str(rec.get("focus", "other")).lower(),
            })
        return cleaned
    except Exception as e:
        print("Failed to parse recommendations JSON:", repr(e))
        # Fallback: plain text bullet lines
        text = resp.choices[0].message.content.strip()
        lines = [
            line.lstrip("-• ").strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("{")
        ]
        if not lines:
            return []
        return [
            {
                "title": lines[i][:80],
                "detail": lines[i],
                "focus": "other",
            }
            for i in range(min(5, len(lines)))
        ]


# =========================================================
# 6. Flask routes
# =========================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("statement")
    if not file or file.filename == "":
        flash("Please upload a PDF bank statement.")
        return redirect(url_for("index"))

    if not file.filename.lower().endswith(".pdf"):
        flash("Only PDF files are supported right now.")
        return redirect(url_for("index"))

    try:
        raw_text = extract_text_from_pdf(file)
        tx_text = extract_transaction_section(raw_text)
        txns = extract_transactions_with_ai(tx_text)
        df = transactions_to_df(txns)

        if df.empty:
            flash("Could not detect any transactions. The AI may need prompt tweaks for this bank format.")
            return redirect(url_for("index"))

        (
            summary,
            monthly_spend,
            category_spend,
            top_categories,
            category_chart,
            monthly_chart,
            other_breakdown,
        ) = compute_insights(df)

        recommendations = generate_recommendations_ai(summary, category_spend, df)

        # Build a transaction list for the chatbot (date, description, amount, type, category)
        df_chat = df.reset_index()[["date", "description", "amount", "type", "category"]]
        # Convert date to string and keep a reasonable number (e.g. 500) to avoid huge payloads
        df_chat["date"] = df_chat["date"].astype(str)
        chat_transactions = df_chat.head(500).to_dict(orient="records")

        # Dashboard context for the chatbot to use
        dashboard = {
            "summary": summary,
            "monthly_spend": monthly_spend,
            "category_spend": category_spend,
            "top_categories": [
                {"category": cat, "amount": amt} for cat, amt in top_categories
            ],
            "other_breakdown": other_breakdown,
            "transactions": chat_transactions,  # 👈 full per-transaction data for chat
        }

        return render_template(
            "result.html",
            summary=summary,
            monthly_spend=monthly_spend,
            category_spend=category_spend,
            top_categories=top_categories,
            category_chart=category_chart,
            monthly_chart=monthly_chart,
            other_breakdown=other_breakdown,
            recommendations=recommendations,
            currency_symbol=CURRENCY_SYMBOL,
            dashboard=dashboard,
        )
    except Exception as e:
        print("Error while analyzing statement:", repr(e))
        flash("Something went wrong while analyzing the statement. Check server logs.")
        return redirect(url_for("index"))


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint for the AI assistant.
    Expects JSON:
      { "question": "...", "dashboard": { ...analytics..., "transactions": [...] } }
    """
    if client is None:
        return jsonify({"error": "AI is not configured on the server."}), 500

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    question = (data.get("question") or "").strip()
    dashboard = data.get("dashboard")

    if not question:
        return jsonify({"error": "Question is required."}), 400
    if not isinstance(dashboard, dict):
        return jsonify({"error": "Dashboard data is missing."}), 400

    # Keep the prompt focused on THIS report but allow pattern analysis
    system_msg = (
        "You are SpendScope AI, an assistant that answers questions about a "
        "single user's bank-statement report.\n\n"
        "You are given a JSON object called 'dashboard_data' with:\n"
        "- summary (totals, period, avg daily spend)\n"
        "- monthly_spend (YYYY-MM -> amount)\n"
        "- category_spend (category -> amount)\n"
        "- top_categories (list of {category, amount})\n"
        "- other_breakdown (what 'Other' includes)\n"
        "- transactions: list of individual transactions with fields:\n"
        "  date (YYYY-MM-DD), description, amount (debits negative, credits positive), "
        "  type (debit/credit), category.\n\n"
        "How to answer:\n"
        "- Use the transactions to answer questions like where shopping happened, "
        "  which merchants are common, largest purchases, recurring charges, etc.\n"
        "- When user asks about 'shopping', look for category == 'Shopping' and quote "
        "  descriptions and amounts.\n"
        "- When user asks for patterns, highlight themes: biggest categories, spikes by month, "
        "  clusters of merchants (e.g., a lot of food delivery), recurring spends.\n"
        "- When useful, reference concrete examples: dates, merchant names from 'description', "
        "  and approximate totals.\n"
        "- You may also give general personal finance tips that are clearly tied to what you see "
        "  in this data (e.g. high spend on food, subscriptions, transport).\n"
        "- If the user asks something that truly is not in the data (e.g. future salary, "
        "  exact bank fees policy), say you don't see that info and explain what you CAN infer.\n"
        "- Answer clearly, in a friendly tone, using 2–6 short paragraphs or bullet lists.\n"
        "- Do NOT output JSON; just plain text."
    )

    user_msg = (
        "User question:\n"
        f"{question}\n\n"
        "Dashboard data (JSON):\n"
        f"{json.dumps(dashboard, indent=2)}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        print("Chat error:", repr(e))
        return jsonify({"error": "There was an error talking to the AI. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True)
