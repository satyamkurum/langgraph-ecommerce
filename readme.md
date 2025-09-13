# E-commerce Customer Support Agent (LangGraph + Streamlit)

This project implements a **conversational customer support assistant** for an e-commerce platform using [LangGraph], [LangChain], [Google Gemini API], and [Streamlit].

The assistant can:
- Answer **FAQs** (e.g., return policy, shipping times, warranty).
- Track **orders by ID**.
- Maintain **conversation history** with persistence.

---

## Features
- **LangGraph Orchestration**: Graph-based control flow for multi-turn conversations.
- **Persistence**: Conversations persist across sessions using SQLite checkpointer.
- **Tool Integration**: LLM can call tools (FAQ, order tracking, human escalation).
- **Structured Parsing**: Order IDs and other fields are parsed with regex + Pydantic or Gemini structured output.
- **Streamlit Frontend**: Simple web chat interface connected to FastAPI backend.

---


## Setup

### 1. Clone the repo
```bash
git clone https://github.com/satyamkurum/ecommerce-langgraph.git
cd ecommerce-langgraph
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate    # (Windows)
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key
```

### 5. Run backend (FastAPI)
```bash
uvicorn src.api.app:app --reload
```

### 6. Run frontend (Streamlit)
```bash
streamlit run streamlit_app.py
```

---

##  Example Tools
- **FAQ Tool** → returns predefined FAQ answers.
- **Track Order Tool** → parses and validates order IDs, returns fake tracking status.
- **Escalation Tool** → routes unresolved queries to a human agent.


##  Testing

```bash
python -m pytest test/quick_test.py -v
```





