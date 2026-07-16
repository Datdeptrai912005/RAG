# RAG + MCP Server cho Robot Xiaozhi

Hệ thống RAG (Retrieval-Augmented Generation) kết hợp MCP (Model Context Protocol), cho phép robot
Xiaozhi tra cứu tri thức nội bộ về chính dự án này (kiến trúc, cách hoạt động, quyết định kỹ thuật)
và trả lời bằng giọng nói.

Xem chi tiết lý thuyết RAG + kiến trúc đầy đủ tại [`knowledge_base/project_overview.md`](knowledge_base/project_overview.md).

## Kiến trúc

```
Robot Xiaozhi (ESP32) → Backend Xiaozhi (ASR/LLM/TTS) → MCP server (project này) → Chroma Vector DB
```

## Cấu trúc thư mục

```
.
├── crawl_data.py           # Tải tài liệu nguồn từ GitHub về knowledge_base/
├── knowledge_base/         # Tài liệu .md dùng làm nguồn tri thức cho RAG
├── src/
│   ├── ingest.py            # Chunk + embed + lưu vào vector_db
│   ├── retriever.py         # Test nhanh similarity search
│   └── generator.py         # Test full RAG chain (retrieval + Gemini) - chỉ dùng để debug
├── knowledge_server.py     # MCP server, expose tool search_knowledge cho backend Xiaozhi gọi
├── requirements.txt
└── .gitignore
```

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Tạo file `.env` ở thư mục gốc (không commit lên Git):

```
GOOGLE_API_KEY=<key lấy tại https://aistudio.google.com/app/apikey>
```

## Cách chạy pipeline

**1. Crawl dữ liệu nguồn (nếu chưa có sẵn trong `knowledge_base/`):**

```bash
python crawl_data.py
```

**2. Nạp vào vector database:**

```bash
python src/ingest.py
```

**3. (Tuỳ chọn) Test retrieval độc lập:**

```bash
python src/retriever.py
```

**4. (Tuỳ chọn) Test full RAG chain với Gemini:**

```bash
python src/generator.py
```

**5. Chạy MCP server:**

```bash
python knowledge_server.py
```

**6. Kết nối với backend Xiaozhi** qua `mcp_pipe.py` (lấy từ [78/mcp-calculator](https://github.com/78/mcp-calculator)):

```bash
export MCP_ENDPOINT=<endpoint lấy từ config agent trên xiaozhi.me>
python mcp_pipe.py knowledge_server.py
```

## Công nghệ sử dụng

| Thành phần         | Công nghệ                                                     |
| ------------------ | ------------------------------------------------------------- |
| Embedding          | `all-MiniLM-L6-v2` (qua `langchain_huggingface`)              |
| Vector DB          | ChromaDB (chạy embedded, local)                               |
| LLM (test độc lập) | Gemini (`gemini-3.1-flash-lite`) qua `langchain_google_genai` |
| Giao thức tool     | MCP (Model Context Protocol), qua `FastMCP`                   |

## Ghi chú

- MCP tool `search_knowledge` chỉ trả về **đoạn tài liệu thô** (không tự sinh câu trả lời), để LLM ở
  backend Xiaozhi tự tổng hợp — giữ đúng pattern chuẩn của `mcp-calculator` và tránh gọi LLM 2 lần.
- `src/generator.py` chỉ dùng để debug/kiểm tra chất lượng retrieval độc lập, không nằm trong pipeline
  chính lúc production.
