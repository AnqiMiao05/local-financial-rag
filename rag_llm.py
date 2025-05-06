import faiss
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import os

class FinanceRAGSystem:
    def __init__(self, index_path, metadata_path, model_path, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_path = Path(model_path)
        self.embedding_model_name = embedding_model_name

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        print(f"Loaded embedding model: {self.embedding_model_name}")

        # Load FAISS index and metadata
        self.load_index_and_metadata()

        # Load LLM
        self.load_llm()

    def load_index_and_metadata(self):
        """Load FAISS index and metadata"""
        print(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"Index contains {self.index.ntotal} vectors")

    def load_llm(self):
        """Load the local LLM model (example with Qwen)"""
        print(f"Loading LLM from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )

        print("LLM loaded successfully")

    def retrieve(self, query, top_k=5):
        """Retrieve relevant document chunks for a query"""
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
        """Generate context string from retrieved results"""
        context_parts = []
        for i, result in enumerate(results):
            metadata = result["metadata"]
            context_parts.append(
                f"[Document {i+1}] {metadata['title']}\n"
                f"{metadata['text']}"
            )
        return "\n\n".join(context_parts)

    def create_prompt(self, query, context):
        """Create a prompt for the LLM"""
        prompt = f"""You are a financial research assistant. Use ONLY the following documents to answer the question. 
If the answer is not present, say "I don't know" and explain why.

Document Content:
{context}

Question: {query}

Answer:"""
        return prompt

    def answer_question(self, query, top_k=5, max_length=512):
        """Answer a question using the RAG approach"""
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "Sorry, I dont have enough info."

        context = self.generate_context(results)
        prompt = self.create_prompt(query, context)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        response = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
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

# ✅ 使用示例
if __name__ == "__main__":
    rag_system = FinanceRAGSystem(
        index_path="data/index/finance_papers.index",
        metadata_path="data/index/finance_papers_metadata.json",
        model_path="./models/mistral-7b-instruct",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Example query
    query = "What are the commonly used risk management strategies in quantitative investment?"
    response = rag_system.answer_question(query)

    print(response["answer"])
    print("\nReferences:")
    for ref in response["references"]:
        print(f"- {ref['title']} (ID: {ref['id']})")
