# run_all.py
import threading
import time
import webbrowser
from pathlib import Path
import uvicorn

def start_server():
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

def open_browser():
    time.sleep(2)
    html_path = Path(__file__).resolve().parent / "rag_web" / "index.html"
    if html_path.exists():
        webbrowser.open("file://" + str(html_path))
    else:
        print(f"Web page file not found: {html_path}")

if __name__ == "__main__":
    print("🚀 Starting the Financial RAG local service and web interface...")
    threading.Thread(target=open_browser).start()
    start_server()