import os
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import random

class VectorIndexBuilder:
    def __init__(self, processed_dir, index_dir, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.processed_dir = Path(processed_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {self.model_name} with dimension {self.embedding_dim}")

    def load_processed_papers(self):
        json_files = list(self.processed_dir.glob("*.json"))
        all_chunks = []
        all_metadata = []

        for json_file in tqdm(json_files, desc="Loading papers"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)

                for chunk in paper_data["chunks"]:
                    all_chunks.append(chunk["text"])
                    all_metadata.append({
                        "chunk_id": chunk["chunk_id"],
                        "paper_id": paper_data["paper_id"],
                        "title": paper_data["metadata"].get("title", ""),
                        "text": chunk["text"]
                    })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        print(f"Loaded {len(all_chunks)} chunks from {len(json_files)} papers")
        return all_chunks, all_metadata

    def create_embeddings(self, chunks):
        embeddings = []
        batch_size = 32

        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
            batch = chunks[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)

        return np.array(embeddings).astype('float32')

    def build_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings)
        print(f"Built FAISS index with {index.ntotal} vectors")
        return index

    def save_index_and_metadata(self, index, metadata):
        index_path = self.index_dir / "finance_papers.index"
        faiss.write_index(index, str(index_path))

        metadata_path = self.index_dir / "finance_papers_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)

        print(f"Saved index to {index_path} and metadata to {metadata_path}")

    def build_index(self):
        chunks, metadata = self.load_processed_papers()
        embeddings = self.create_embeddings(chunks)
        index = self.build_faiss_index(embeddings)
        self.save_index_and_metadata(index, metadata)

        # Output example
        sample_idx = random.randint(0, len(metadata) - 1)
        print("\n Sample metadata:")
        print(json.dumps(metadata[sample_idx], indent=2, ensure_ascii=False))
        print(" Sample embedding shape:", embeddings[sample_idx].shape)

        return index, metadata

if __name__ == "__main__":
    index_builder = VectorIndexBuilder(
        processed_dir="data/processed",
        index_dir="data/index",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    index, metadata = index_builder.build_index()
