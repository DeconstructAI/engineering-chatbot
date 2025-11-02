import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode, disable GPU for Streamlit Cloud

import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import pickle
import PyPDF2
import openai

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Engineering PDF Chatbot", page_icon="⚙️")
st.title("⚙️ Engineering Knowledge Chatbot (Persistent Version)")
st.write("Upload PDFs once, and they’ll be remembered permanently!")

# --- File Paths ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
EMBED_FILE = os.path.join(DATA_DIR, "embeddings.index")
TEXT_FILE = os.path.join(DATA_DIR, "texts.pkl")

# --- Load Embedding Model (CPU only) ---
st.write("🔄 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.to(torch.device("cpu"))

# --- OpenAI API Key (from Streamlit Secrets) ---
# Make sure your Streamlit secrets contain:
# [openai]
# api_key = "your-openai-api-key"
openai.api_key = st.secrets["openai"]["api_key"]

# --- Load Existing Data ---
def load_existing_data():
    if os.path.exists(EMBED_FILE) and os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, "rb") as f:
            texts = pickle.load(f)
        index = faiss.read_index(EMBED_FILE)
        st.session_state["knowledge"] = texts
        st.session_state["index"] = index
        st.success(f"✅ Loaded {len(texts)} chunks from previous session.")
    else:
        st.session_state["knowledge"] = []
        st.session_state["index"] = None

# --- Save Data ---
def save_data():
    with open(TEXT_FILE, "wb") as f:
        pickle.dump(st.session_state["knowledge"], f)
    faiss.write_index(st.session_state["index"], EMBED_FILE)
    st.success("💾 Knowledge base saved permanently!")

# --- Extract Text from PDF ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- Retrieve Most Relevant Chunks ---
def retrieve_chunks(query, k=3):
    if st.session_state["index"] is None:
        return ["No knowledge base found."]
    query_vec = model.encode([query]).astype("float32")
    distances, indices = st.session_state["index"].search(query_vec, k)
    return [st.session_state["knowledge"][i] for i in indices[0]]

# --- Initialize Memory ---
if "knowledge" not in st.session_state:
    load_existing_data()

# --- PDF Upload Section ---
uploaded_files = st.file_uploader(
    "📄 Upload new PDF documents to add to your permanent knowledge base:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    new_texts = []
    for pdf in uploaded_files:
        text = extract_text_from_pdf(pdf)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        new_texts.extend(chunks)

    st.session_state["knowledge"].extend(new_texts)

    # Create new embeddings
    st.write("🔄 Creating embeddings for new files...")
    new_embeddings = model.encode(new_texts, show_progress_bar=True).astype("float32")

    if st.session_state["index"] is None:
        index = faiss.IndexFlatL2(new_embeddings.shape[1])
        index.add(new_embeddings)
        st.session_state["index"] = index
    else:
        st.session_state["index"].add(new_embeddings)

    save_data()

# --- Chat Interface ---
if user_query := st.chat_input("Ask a question about your engineering PDFs..."):
    st.chat_message("user").write(user_query)

    relevant_chunks = retrieve_chunks(user_query)
    context = "\n\n".join(relevant_chunks)

    try:
        # ✅ Compatible with openai==0.28.0
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert engineering assistant. Use the provided context to answer accurately."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"
                }
            ],
            max_tokens=1000,
            temperature=0.2,
        )

        answer = response["choices"][0]["message"]["content"]
        st.chat_message("assistant").write(answer)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
