import fitz  # PyMuPDF
import os
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FinancialPaperProcessor:
    def __init__(self, papers_dir, output_dir):
        self.papers_dir = Path(papers_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.failed_files = []

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extract PDF text and meta information using PyMuPDF"""
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
            print(f"Error processing {pdf_path}: {e}")
            self.failed_files.append(str(pdf_path))
            return "", {}

    def merge_external_metadata(self, pdf_path, metadata):
        """Merge the .json meta information from the download phase (if present)"""
        meta_json_path = pdf_path.with_suffix(".json")
        if meta_json_path.exists():
            try:
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    external_meta = json.load(f)
                metadata.update(external_meta)
            except Exception as e:
                print(f"Failed to merge metadata for {pdf_path}: {e}")
        return metadata

    def process_paper(self, pdf_path):
        """Processing a single paper: extracting text, generating chunks, and saving structured JSON"""
        paper_id = pdf_path.stem
        print(f"Processing paper: {paper_id}")

        text_content, metadata = self.extract_text_from_pdf(pdf_path)
        if not text_content:
            return None

        metadata = self.merge_external_metadata(pdf_path, metadata)

        # Generate chunks
        chunks = self.text_splitter.split_text(text_content)
        print(f"Created {len(chunks)} chunks")

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
        """Batch process all PDF files in a directory (support subdirectories)"""
        pdf_files = list(self.papers_dir.rglob("*.pdf"))
        print(f"Found {len(pdf_files)} papers to process")

        processed_count = 0
        for pdf_file in pdf_files:
            result = self.process_paper(pdf_file)
            if result:
                processed_count += 1

        print(f"Successfully processed {processed_count} papers")
        if self.failed_files:
            print(f"Failed to process {len(self.failed_files)} papers:")
            for f in self.failed_files:
                print(f"   - {f}")

if __name__ == "__main__":
    processor = FinancialPaperProcessor(
        papers_dir="data/papers",
        output_dir="data/processed"
    )
    processor.process_all_papers()
