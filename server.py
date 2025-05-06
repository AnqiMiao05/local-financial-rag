# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_llm import FinanceRAGSystem

app = FastAPI()

# 跨域配置，允许本地网页访问 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化本地 RAG 系统
rag = FinanceRAGSystem(
    index_path="data/index/finance_papers.index",
    metadata_path="data/index/finance_papers_metadata.json",
    model_path="models/mistral-7b-instruct",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 请求数据结构
class QueryRequest(BaseModel):
    query: str

# 接口定义
@app.post("/query")
def query_rag(request: QueryRequest):
    return rag.answer_question(request.query)