# Intelligent Vehicle Specification Extraction System

A Streamlit application that extracts vehicle specifications from service-manual PDFs.

## Stack

- Streamlit
- LangChain
- FAISS
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Groq Llama 3.3
- PyMuPDF

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the Groq key:

```bash
export GROQ_API_KEY="your-key"
```

Run:

```bash
streamlit run app.py
```

## Streamlit Cloud

Add this secret in the app settings:

```toml
GROQ_API_KEY = "your-key"
```

Then upload a service-manual PDF through the application.
