import arxiv
import time
from pathlib import Path
import os
import json
from tqdm import tqdm

# Define the q-fin categories
q_fin_categories = [
    "q-fin.CP",  # Computational Finance
    "q-fin.EC",  # Economics
    "q-fin.GN",  # General Finance
    "q-fin.MF",  # Mathematical Finance
    "q-fin.PM",  # Portfolio Management
]

# Set up the root download directory
download_root = Path("data/papers")
download_root.mkdir(parents=True, exist_ok=True)

client = arxiv.Client(
    page_size=100,
    delay_seconds=3.0,
    num_retries=3
)

for category in q_fin_categories:
    print(f"\nProcessing category: {category}")
    
    category_dir = download_root / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Create search query
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # Iterate over results with progress bar
    for result in tqdm(client.results(search), desc=f"{category}"):
        paper_id = result.get_short_id()
        filename = f"{paper_id}.pdf"
        filepath = category_dir / filename
        meta_path = category_dir / f"{paper_id}.json"

        if filepath.exists():
            continue

        try:
            print(f"Downloading {paper_id}: {result.title}")
            result.download_pdf(dirpath=str(category_dir), filename=filename)
            
            meta = {
                "id": paper_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": str(result.published),
                "summary": result.summary,
                "category": category,
                "pdf_path": str(filepath)
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                
            time.sleep(1)
            
        except Exception as e:
            print(f"Failed to download {paper_id}: {e}")
            continue

    print(f" Completed category: {category}")

print("Download process completed.")
