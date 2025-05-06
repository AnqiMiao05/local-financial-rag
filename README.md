# 📘 Financial RAG Assistant

A local, privacy-preserving **Retrieval-Augmented Generation (RAG)** system tailored for **financial paper understanding and question answering**, built with:

- 🧾 arXiv PDF scraping & preprocessing
- 🔍 FAISS-based vector retrieval
- 🤖 Local LLM inference (Mistral-7B)
- 🌐 A modern HTML/JS front-end for local interaction

---

## 🚀 Features

- Download and process over 1000+ arXiv quantitative finance papers
- Chunk and embed texts with SentenceTransformers
- Build fast semantic search using FAISS
- Load 4-bit quantized LLM (Mistral-7B) via HuggingFace & Transformers
- Ask financial questions via a sleek web interface (with history)

---

## 📦 Project Structure

```bash
RAG/
├── data/                  # Downloaded PDFs & processed chunks
│   ├── papers/            # Raw PDFs
│   ├── processed/         # JSON chunks
│   └── index/             # FAISS index + metadata
├── models/                # Local model (Mistral-7B GGUF)
├── rag_web/               # Frontend UI
│   ├── index.html         # Simple web interface
│   └── app.js             # JS interaction logic
├── data_collection.py     # Arxiv scraping
├── txt_processing.py      # PDF parsing & chunking
├── vector_indexing.py     # FAISS index builder
├── rag_llm.py             # LLM + Retriever + Answer engine
├── server.py              # FastAPI backend
└── run_all.py             # One-click run script
```

---

## 🔧 Environment Setup

```bash
# 1. Create repo
Copy all code files to your project folder

# 2. Create conda env (recommended)
conda create -n rag python=3.10 -y
conda activate rag

# 3. Install dependencies
```

### 🧱 Key dependencies

| Package               | Version |
| --------------------- | ------- |
| torch                 | >=2.1   |
| transformers          | >=4.36  |
| sentence-transformers | >=2.3   |
| faiss-cpu             | >=1.7   |
| PyMuPDF               | >=1.22  |
| uvicorn               | >=0.20  |
| fastapi               | >=0.95  |
| bitsandbytes          | >=0.41  |
| accelerate            | >=0.26  |
| numpy                 | ==1.24.4|
| scipy                 | ==1.11.4|
| longchain             | ==0.3.23|
| tqbm                  | ==4.67.1|
| huggingface-hub       | ==0.30.1|
---

## 🤗 Download Mistral-7B Locally

This project uses **Mistral-7B-Instruct v0.2** from HuggingFace Hub. You need to download it manually:

### 1. Register on Hugging Face

Go to [https://huggingface.co/join](https://huggingface.co/join) and create an account.

### 2. Create Access Token

- Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Create a token (recommended scope: `read`)
- Make sure to check 'Read access to contents of all public gated repos you can access'

### 3. Log in via CLI

```bash
huggingface-cli login
```

Paste your token when prompted.

### 4. Download model

```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir ./models/mistral-7b-instruct --local-dir-use-symlinks False
```

>  Tip: Total \~25GB, make sure you have enough space.

---

##  Preprocessing Pipeline

1. **Download papers**

```bash
python data_collection.py
```
After running this script, './data' will be generated automatically.

2. **Extract and chunk PDFs**

```bash
python txt_processing.py
```

3. **Build FAISS index**

```bash
python vector_indexing.py
```

---

##  Run Locally

```bash
python run_all.py
```

This will:

- Launch FastAPI server at `localhost:8000`
- Open your browser to a local web UI

You can ask questions like:

```text
"What is Value at Risk and how is it calculated?"
"What are popular risk management frameworks in portfolio theory?"
```

---

##  Updating Data & Custom PDF Import

###  Increase the Number of Downloads (No Need to Clear Old Data)

To fetch more papers from arXiv, simply increase the `max_results` value in `data_collection.py`:

```python
max_results = 1000  # Or any desired number
```

The system will automatically skip PDFs that already exist (based on filename), so there's **no need to delete existing data** when performing incremental updates.

---

###  Custom PDF Upload (Manual Import)

You can also manually add any financial-related PDFs to the following folder:

```bash
data/papers/
```

For example: `data/papers/fama_french_1993.pdf`

Then run the following scripts to integrate the new documents into the knowledge base:

```bash
python txt_processing.py        # Extract text and chunk into JSON
python vector_indexing.py      # Generate embeddings and build FAISS index
```

This allows you to seamlessly add and update your personal knowledge base.

---

### 💡 Notes:

- English-language financial PDFs are recommended
- Avoid using Chinese characters or spaces in filenames
- The system automatically skips already-processed documents (only new ones are handled)

---

## Notes

- LLM loads via 4-bit quantization for memory efficiency (tested on RTX 4060)
- Works offline after model download
- Supports interactive web UI with query history + citation tracing
- You can use any models for vector indexing and LLM, based on your requirements and hardware.
