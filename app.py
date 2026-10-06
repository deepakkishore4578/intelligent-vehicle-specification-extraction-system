import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Mechanic AI - Service Hub",
    page_icon="🚗",
    layout="wide",
)


# ============================================================
# Configuration
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

PRESET_QUERIES = [
    "Torque specifications for front suspension",
    "Torque specifications for braking system",
    "Service materials",
    "Wheel alignment specifications",
    "Grease and lubricants",
]


# ============================================================
# Helpers
# ============================================================
def get_groq_api_key():
    """Read the Groq key from Streamlit Secrets or environment."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None

    return key or os.environ.get("GROQ_API_KEY")


def extract_json_from_text(text):
    """Extract a JSON object from the model response."""
    try:
        text = str(text).replace("```json", "").replace("```", "")
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            return json.loads(match.group(0))

        return json.loads(text)

    except Exception:
        return {"specs": []}


@st.cache_resource
def load_embeddings():
    """Load the local Sentence Transformer embedding model once."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def build_vector_store(pdf_bytes, file_hash):
    """
    Build the FAISS vector store from the uploaded service manual.

    The file hash is included so Streamlit creates a new cached index
    whenever the uploaded PDF changes.
    """
    del file_hash  # Used only as part of the cache key.

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(documents)

        embeddings = load_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)

        return vector_store, len(documents), len(chunks)

    finally:
        Path(pdf_path).unlink(missing_ok=True)


PROMPT_TEMPLATE = """
You are a highly accurate technical data extractor.

Your task is to analyze the provided service-manual context and extract
ALL specifications related to the user's query.

CRITICAL INSTRUCTIONS:

1. The text may contain long tables (20+ rows). Extract EVERY relevant row.
2. Torque tables typically list values in this order:
   Nm -> lb-ft -> lb-in.
3. If you see a pattern such as "12 — 106", interpret:
   - 12 as Nm
   - — means no lb-ft value
   - 106 as lb-in
4. "value" MUST contain only the value, without the unit.
5. "unit" MUST contain only the unit code, such as Nm, lb-ft, lb-in, L, etc.
6. For parts/specifications without a unit, use null or an empty string.
7. Do not invent information that is not present in the context.

Return ONLY JSON in this format:

{{
    "specs": [
        {{
            "component": "part name",
            "spec_type": "Torque",
            "value": "17",
            "unit": "Nm"
        }}
    ]
}}

If no relevant data is found, return:

{{"specs": []}}

Context:
{context}

Query:
{question}
"""


def extract_specs(vector_store, question, api_key):
    """Retrieve relevant context and extract structured specifications."""
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGroq(
        temperature=0,
        model_name=GROQ_MODEL,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    data = extract_json_from_text(response.content)
    return data.get("specs", [])


# ============================================================
# UI
# ============================================================
st.markdown(
    """
    <style>
        .mechanic-header {
            background: #2C3E50;
            color: white;
            padding: 18px 22px;
            border-radius: 10px;
            margin-bottom: 18px;
        }

        .mechanic-header h1 {
            margin: 0;
            font-size: 28px;
        }

        .mechanic-subheader {
            margin-top: 5px;
            opacity: 0.8;
        }
    </style>

    <div class="mechanic-header">
        <h1>🚗 Mechanic AI: Service Hub</h1>
        <div class="mechanic-subheader">
            Intelligent vehicle specification extraction powered by Groq
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = get_groq_api_key()

if not api_key:
    st.error(
        "GROQ_API_KEY is not configured. Add it in Streamlit → Settings → Secrets."
    )
    st.stop()


# ============================================================
# PDF input
# ============================================================
st.subheader("1. Upload Service Manual")

uploaded_pdf = st.file_uploader(
    "Upload the vehicle service manual PDF",
    type=["pdf"],
)

if uploaded_pdf is None:
    st.info("Upload a service manual PDF to begin.")
    st.stop()


# ============================================================
# Build / cache vector store
# ============================================================
pdf_bytes = uploaded_pdf.getvalue()
file_hash = hashlib.sha256(pdf_bytes).hexdigest()

with st.spinner("Reading PDF and preparing the search index..."):
    vector_store, page_count, chunk_count = build_vector_store(
        pdf_bytes,
        file_hash,
    )

st.success(
    f"Service manual ready — {page_count} pages indexed into {chunk_count} chunks."
)


# ============================================================
# Query UI
# ============================================================
st.subheader("2. Extract Specifications")

selected_query = st.selectbox(
    "Quick Select",
    ["--- Select a Quick Query ---"] + PRESET_QUERIES,
)

custom_query = st.text_input(
    "Custom Query",
    placeholder="e.g. What is the torque specification for the front shock absorber?",
)

if selected_query != "--- Select a Quick Query ---":
    query = selected_query
else:
    query = custom_query.strip()


if st.button("🔍 Extract Data", type="primary"):
    if not query:
        st.warning("Please select a quick query or enter a custom query.")
        st.stop()

    with st.spinner("Searching the manual and extracting specifications..."):
        try:
            specs = extract_specs(
                vector_store=vector_store,
                question=query,
                api_key=api_key,
            )
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
            st.stop()

    if specs:
        st.success(f"Found {len(specs)} specifications.")

        st.dataframe(
            specs,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps({"specs": specs}, indent=4),
            file_name="vehicle_specs.json",
            mime="application/json",
        )
    else:
        st.warning(f"No relevant specifications found for: {query}")
