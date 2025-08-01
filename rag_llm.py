import faiss
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import os
from config import config
from logger import logger

class FinanceRAGSystem:
    def __init__(self, index_path=None, metadata_path=None, model_path=None, embedding_model_name=None):
        # 使用配置中的路径，如果没有提供参数
        self.index_path = Path(index_path or config.get("data.index_file"))
        self.metadata_path = Path(metadata_path or config.get("data.metadata_file"))
        self.model_path = Path(model_path or config.get("models.llm_path"))
        self.embedding_model_name = embedding_model_name or config.get("models.embedding_model")

        # 从配置获取设备设置
        device_config = config.get("models.device", "auto")
        if device_config == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_config
        logger.info(f"Using device: {self.device}")

        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        logger.info(f"Loaded embedding model: {self.embedding_model_name}")

        # Load FAISS index and metadata
        self.load_index_and_metadata()

        # Load LLM
        self.load_llm()

    def load_index_and_metadata(self):
        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        logger.info(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        logger.info(f"Index contains {self.index.ntotal} vectors")

    def load_llm(self):
        logger.info(f"Loading LLM from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # 从配置获取模型参数
        load_in_4bit = config.get("models.load_in_4bit", True)
        torch_dtype = getattr(torch, config.get("models.torch_dtype", "float16"))

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
        )

        logger.info("LLM loaded successfully")

    def retrieve(self, query, top_k=None):
        # 从配置获取默认的top_k值
        if top_k is None:
            top_k = config.get("rag.top_k", 5)
            
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "score": float(distances[0][i]),
                    "metadata": self.metadata[idx]
                })

        return results

    def generate_context(self, results):
        context_parts = []
        for i, result in enumerate(results):
            metadata = result["metadata"]
            context_parts.append(
                f"[Document {i+1}] {metadata['title']}\n"
                f"{metadata['text']}"
            )
        return "\n\n".join(context_parts)

    def create_prompt(self, query, context):
        prompt = f"""You are a financial research assistant. Use ONLY the following documents to answer the question. 
If the answer is not present, say "I don't know" and explain why.

Document Content:
{context}

Question: {query}

Answer:"""
        return prompt

    def answer_question(self, query, top_k=None, max_length=None):
        # 从配置获取默认参数
        if top_k is None:
            top_k = config.get("rag.top_k", 5)
        if max_length is None:
            max_length = config.get("rag.max_new_tokens", 1024)
            
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "Sorry, I dont have enough info."

        context = self.generate_context(results)
        prompt = self.create_prompt(query, context)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 从配置获取生成参数
        temperature = config.get("rag.temperature", 0.7)
        top_p = config.get("rag.top_p", 0.9)
        repetition_penalty = config.get("rag.repetition_penalty", 1.1)
        
        response = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty
        )

        answer = self.tokenizer.decode(response[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        references = []
        for result in results:
            metadata = result["metadata"]
            if metadata["paper_id"] not in [ref["id"] for ref in references]:
                references.append({
                    "id": metadata["paper_id"],
                    "title": metadata["title"]
                })

        return {
            "answer": answer.strip(),
            "references": references
        }

# Example Usage
if __name__ == "__main__":
    rag_system = FinanceRAGSystem()

    # Example query
    query = "What are the commonly used risk management strategies in quantitative investment?"
    response = rag_system.answer_question(query)

    logger.info(response["answer"])
    logger.info("\nReferences:")
    for ref in response["references"]:
        logger.info(f"- {ref['title']} (ID: {ref['id']})")
