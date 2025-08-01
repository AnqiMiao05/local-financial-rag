# run_all.py
import threading
import time
import webbrowser
from pathlib import Path
import uvicorn
from config import config
from logger import logger

def start_server():
    host = config.get("server.host", "127.0.0.1")
    port = config.get("server.port", 8000)
    reload = config.get("server.reload", True)
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=reload)

def open_browser():
    time.sleep(2)
    html_path = Path(__file__).resolve().parent / "rag_web" / "index.html"
    if html_path.exists():
        webbrowser.open("file://" + str(html_path))
        logger.info("Opened web interface in browser")
    else:
        logger.error(f"Web page file not found: {html_path}")

if __name__ == "__main__":
    logger.info("Starting the Financial RAG local service and web interface...")
    config.create_directories()
    threading.Thread(target=open_browser).start()
    start_server()
