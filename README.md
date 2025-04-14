# 🛡️ Health Insurance RAG QA Chatbot

A Streamlit chatbot built using Retrieval-Augmented Generation (RAG) to answer queries about health insurance documents (ICICI, HDFC).

## ✅ Features
- Multi-turn chat
- Source-cited answers with page numbers
- History tab to review past interactions
- Google Gemini + FAISS + LangChain

## 🚀 Setup Instructions

1. Clone or unzip this project
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Create a `.env` file:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```
4. Run the app:
    ```bash
    streamlit run app.py
    ```

Place your PDFs in the same directory:
- `HDFC Optima Restore.pdf`
- `ICICI max-protect.pdf`

---

## 🔐 Security
FAISS index loading uses `allow_dangerous_deserialization=True`. Do **NOT** load untrusted indexes.