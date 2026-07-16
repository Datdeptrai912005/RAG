import os
import time
import requests

BASE_URL = "https://raw.githubusercontent.com/78/xiaozhi-esp32/main"
TARGET_DIR = "./knowledge_base"
FILES_TO_CRAWL = [
    "README.md",
    "docs/custom-board.md",
    "docs/websocket.md",
    "partitions/v2/README.md",
    "main/boards/freenove-esp32s3-display-2.8-lcd/ReadMe.md"
]

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

def download_file(path):
    url = f"{BASE_URL}/{path}"
    response = requests.get(url)
    if response.status_code == 200:
        filename = path.replace("/", "_")
        with open(os.path.join(TARGET_DIR, filename), "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Đã tải thành công: {filename}")
    else:
        print(f"Lỗi tải {path}: {response.status_code}")

for file_path in FILES_TO_CRAWL:
    download_file(file_path)
    time.sleep(3)