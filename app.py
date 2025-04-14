import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0.2
)

pdf_paths = ['HDFC Optima Restore.pdf', 'ICICI max-protect.pdf']
all_docs = []
for path in pdf_paths:
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata['source'] = path
        doc.metadata['page'] = doc.metadata.get("page", "N/A")
    all_docs.extend(docs)


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunked_docs = splitter.split_documents(all_docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


if os.path.exists("insurance_faiss_index"):
    vectorstore = FAISS.load_local(
        "insurance_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    vectorstore.save_local("insurance_faiss_index")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "fetch_k": 20})


category_traits = {
    "Coverage & Benefits": "Enumerate what's covered, mention exclusions, co-pay clauses, and cite specific clauses or page numbers.",
    "Limits & Sub-limits": "Provide exact figures and sub-limits with references to specific clauses or plan variations.",
    "Eligibility & Waiting Periods": "Mention exact waiting periods, relevant conditions or exceptions, and cite specific policy pages.",
    "Network & Claims Process": "Detail the claim filing process, network hospitals, timelines, and reference procedural clauses.",
    "Others": "Answer comprehensively and cite document pages/clauses if applicable."
}

def classify_question_category(query):
    q = query.lower()
    if any(word in q for word in ["coverage", "covered", "benefit", "domiciliary"]):
        return "Coverage & Benefits"
    elif any(word in q for word in ["co-pay", "limit", "sub-limit", "room rent"]):
        return "Limits & Sub-limits"
    elif any(word in q for word in ["waiting", "eligibility", "pre-existing", "period"]):
        return "Eligibility & Waiting Periods"
    elif any(word in q for word in ["claim", "cashless", "network", "hospital"]):
        return "Network & Claims Process"
    else:
        return "Others"


custom_prompt_template = PromptTemplate.from_template("""
You are a helpful health insurance expert. Use only the provided context to answer the user's question.

Question: {question}

Context:
{context}

{instruction}

Provide a detailed answer. Cite exact page numbers or clause references.
**Explain your reasoning step-by-step before providing the final answer. You MUST cite the source document and page number/clause.**
Answer:
""")


qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": custom_prompt_template,
        "document_variable_name": "context"
    },
    return_source_documents = True
)


st.set_page_config(page_title="Q/A Insurance", layout="wide")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "view" not in st.session_state:
    st.session_state.view = "chat"


with st.sidebar:
    st.markdown("## 📚 Navigation")
    if st.button("🔄 Back to Chat"):
        st.session_state.view = "chat"
    if st.button("🕘 View History"):
        st.session_state.view = "history"
    st.markdown("---")
    if st.button("🧹 Clear History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")


if st.session_state.view == "chat":
    st.title("🛡️ Health Insurance RAG Chat")
    user_query = st.text_input("💬 Ask your question")

    if user_query:
        category = classify_question_category(user_query)
        instruction = category_traits[category]

        result = qa_chain.invoke({
            "question": user_query,
            "instruction": instruction
        })

        answer = result["answer"]
        sources = []
        seen = set()
        for doc in result["source_documents"]:
            src = f"{doc.metadata['source']} - Page {doc.metadata.get('page', '?')}"
            if src not in seen:
                sources.append(src)
                seen.add(src)

        st.session_state.chat_history.append({
            "question": user_query,
            "answer": answer,
            "sources": sources,
            "docs": result["source_documents"]
        })


    for turn in st.session_state.chat_history[::-1]:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown("### ✅ Answer")
            st.write(turn["answer"])
            st.markdown("### 📚 Sources")
            for s in turn["sources"]:
                st.write("📄", s)

elif st.session_state.view == "history":
    st.title("📜 Full Chat History")
    if not st.session_state.chat_history:
        st.info("No history yet. Ask a question first.")
    else:
        for i, turn in enumerate(st.session_state.chat_history):
            with st.expander(f"🗂️ Q{i+1}: {turn['question'][:60]}..."):
                st.markdown("**Q:** " + turn["question"])
                st.markdown("**A:** " + turn["answer"])
                st.markdown("**📚 Sources:**")
                for src in turn["sources"]:
                    st.write("📄", src)

