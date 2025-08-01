import arxiv
import time
from pathlib import Path
import os
import json
from tqdm import tqdm
from config import config
from logger import logger

# 创建必要的目录
config.create_directories()

# 获取配置
q_fin_categories = config.get("arxiv_categories")
download_root = Path(config.get("data.papers_dir"))
max_results = config.get("processing.max_results_per_category", 100)
download_delay = config.get("processing.download_delay", 1.0)

logger.info(f"Starting data collection with {len(q_fin_categories)} categories")
logger.info(f"Download directory: {download_root}")
logger.info(f"Max results per category: {max_results}")

client = arxiv.Client(
    page_size=config.get("processing.arxiv_page_size", 100),
    delay_seconds=config.get("processing.arxiv_delay_seconds", 3.0),
    num_retries=config.get("processing.arxiv_num_retries", 3)
)

for category in q_fin_categories:
    logger.info(f"Processing category: {category}")
    
    category_dir = download_root / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Create search query
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
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
            logger.info(f"Downloading {paper_id}: {result.title}")
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
                
            time.sleep(download_delay)
            
        except Exception as e:
            logger.error(f"Failed to download {paper_id}: {e}")
            continue

    logger.info(f"Completed category: {category}")

logger.info("Download process completed.")
