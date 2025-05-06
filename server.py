# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_llm import FinanceRAGSystem

app = FastAPI()

# Allow local web pages to access APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = FinanceRAGSystem(
    index_path="data/index/finance_papers.index",
    metadata_path="data/index/finance_papers_metadata.json",
    model_path="models/mistral-7b-instruct",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest):
    return rag.answer_question(request.query)
