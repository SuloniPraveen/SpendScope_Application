# 🧭 SpendScope – AI-Powered Spending Insights & Chatbot Advisor

SpendScope is an intelligent personal finance tool that reads bank statements, extracts transaction details, visualizes spending patterns, and generates personalized financial insights through an AI chatbot. It combines **PDF parsing**, **data analytics**, **interactive visualizations**, and an **AI recommendation engine** into one seamless application—helping users understand their financial behavior instantly.

[![▶ Watch Demo](Thumbnail.png)](https://drive.google.com/file/d/1LaEJCW2ZQbzy98E2K03e5OtBvz5ZIuKa/view?usp=share_link)

---

## ✨ Key Features

### 🔍 Automated Statement Parsing

- Upload any PDF bank statement
- System extracts dates, amounts, categories, balance trends, and recurring expenses

### 📊 Spending Visualizations

- Bar graphs, line charts, heatmaps
- Category-wise spending breakdown
- Cash flow trend analysis
- Highest spenders / recurring debit detection

### 🤖 Smart AI Chatbot

- Get personalized recommendations
- Ask questions like _“Where do I spend the most?”_ or _“How can I save more?”_
- AI summarizes your finances in simple, easy-to-read language

### 📈 Behavioral Insights

- Detects overspending patterns
- Identifies recurring bills & EMIs
- Highlights financial risks
- Provides actionable tips

### ⏳ Progress Tracker

- Real-time progress stages
- Smooth transitions
- Displays _“Loading… please wait”_ until analysis completes

---

# 🛠 Tech Stack

- **Python**
- **Flask**
- **pdfplumber**
- **pandas**
- **matplotlib**
- **HTML + CSS**
- **OpenAI API**

---

# 📥 Installation & Setup Guide

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/spendscope.git
cd spendscope
```

## 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Sample requirements:

```
flask
pdfplumber
pandas
matplotlib
python-dotenv
openai
```

## 4. Add OpenAI API Key

Create `.env`:

```
OPENAI_API_KEY=your_api_key_here
```

## 5. Run the Application

```bash
python3 app.py
```

## 6. Open in Browser

```
http://127.0.0.1:5000
```

---

# 🚀 Why SpendScope is Useful

- Saves time analyzing bank statements
- Clear visual summaries
- Personalized AI financial guidance
- Detects risky spending behavior
- Provides clear monthly insights
- Helps with budgeting, audits, and planning

---

# 📌 Future Enhancements

- Multi-bank categorization
- PDF/Excel exportable reports
- Savings goal predictor
- Mobile UI version

---
