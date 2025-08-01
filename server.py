# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_llm import FinanceRAGSystem
from config import config
from logger import logger

app = FastAPI(title="Financial RAG API", version="1.0.0")

# Allow local web pages to access APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("server.cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化RAG系统
logger.info("Initializing RAG system...")
rag = FinanceRAGSystem(
    index_path=config.get("data.index_file"),
    metadata_path=config.get("data.metadata_file"),
    model_path=config.get("models.llm_path"),
    embedding_model_name=config.get("models.embedding_model")
)
logger.info("RAG system initialized successfully")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    try:
        result = rag.answer_question(request.query)
        logger.info("Query processed successfully")
        return result
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "RAG system is running"}
