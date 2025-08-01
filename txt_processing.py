import fitz  # PyMuPDF
import os
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import config
from logger import logger

class FinancialPaperProcessor:
    def __init__(self, papers_dir=None, output_dir=None):
        # 使用配置中的路径，如果没有提供参数
        self.papers_dir = Path(papers_dir or config.get("data.papers_dir"))
        self.output_dir = Path(output_dir or config.get("data.processed_dir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.failed_files = []

        # 从配置获取分块参数
        chunk_size = config.get("processing.chunk_size", 1000)
        chunk_overlap = config.get("processing.chunk_overlap", 200)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        logger.info(f"Initialized processor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def extract_text_from_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text_content = ""
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
            }

            for page in doc:
                text = page.get_text("text")

                # Skip license / cover page
                if "arXiv" in text or "license" in text.lower():
                    continue

                text_content += text + "\n\n"

            return text_content.strip(), metadata

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            self.failed_files.append(str(pdf_path))
            return "", {}

    def merge_external_metadata(self, pdf_path, metadata):
        meta_json_path = pdf_path.with_suffix(".json")
        if meta_json_path.exists():
            try:
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    external_meta = json.load(f)
                metadata.update(external_meta)
            except Exception as e:
                logger.error(f"Failed to merge metadata for {pdf_path}: {e}")
        return metadata

    def process_paper(self, pdf_path):
        paper_id = pdf_path.stem
        logger.info(f"Processing paper: {paper_id}")

        text_content, metadata = self.extract_text_from_pdf(pdf_path)
        if not text_content:
            return None

        metadata = self.merge_external_metadata(pdf_path, metadata)

        # Generate chunks
        chunks = self.text_splitter.split_text(text_content)
        logger.info(f"Created {len(chunks)} chunks")

        document = {
            "paper_id": paper_id,
            "metadata": metadata,
            "chunks": [
                {
                    "chunk_id": f"{paper_id}_chunk_{i}",
                    "text": chunk,
                    "source": paper_id
                } for i, chunk in enumerate(chunks)
            ]
        }

        output_path = self.output_dir / f"{paper_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)

        return document

    def process_all_papers(self):
        pdf_files = list(self.papers_dir.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} papers to process")

        processed_count = 0
        for pdf_file in pdf_files:
            result = self.process_paper(pdf_file)
            if result:
                processed_count += 1

        logger.info(f"Successfully processed {processed_count} papers")
        if self.failed_files:
            logger.warning(f"Failed to process {len(self.failed_files)} papers:")
            for f in self.failed_files:
                logger.warning(f"   - {f}")

if __name__ == "__main__":
    processor = FinancialPaperProcessor()
    processor.process_all_papers()
