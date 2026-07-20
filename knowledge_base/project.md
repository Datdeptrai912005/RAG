# Tổng Quan Dự Án & Cơ Sở Lý Thuyết RAG

## 1. Mục Tiêu Của Đề Tài Nghiên Cứu

Mục tiêu cốt lõi của đề tài là xây dựng một MCP Server cung cấp khả năng tra cứu tri thức (RAG - Retrieval-Augmented Generation) cho robot Xiaozhi.

Khi người dùng đặt câu hỏi liên quan đến dự án (như kiến trúc hệ thống, cách thức hoạt động, hoặc các quyết định kỹ thuật), robot sẽ gọi công cụ `search_knowledge` để tìm kiếm đoạn thông tin liên quan nhất trong kho tri thức nội bộ. Sau đó, LLM sẽ sử dụng kết quả truy xuất được để tổng hợp và trả lời người dùng bằng giọng nói một cách chính xác.

## 2. Khái Niệm Tổng Quan Về RAG

RAG (Retrieval-Augmented Generation) là một kỹ thuật tiên tiến giúp nâng cao khả năng truy xuất và sinh văn bản của Mô hình ngôn ngữ lớn (LLM) bằng cách kết hợp với các nguồn tài liệu tri thức bên ngoài.

### 2.1. Tại sao RAG lại quan trọng?

RAG đóng vai trò thiết yếu trong việc phát triển các chatbot thông minh và ứng dụng Xử lý ngôn ngữ tự nhiên (NLP). Kỹ thuật này giúp giải quyết những thách thức lớn nhất của các LLM hiện nay:

- **Giảm thiểu ảo giác (Hallucinations):** LLM thường có xu hướng trình bày thông tin sai lệch khi thiếu dữ liệu hoặc bị ép trả lời các vấn đề ngoài phạm vi huấn luyện.
- **Cập nhật thông tin:** LLM thường bị giới hạn bởi thời điểm cắt dữ liệu huấn luyện (knowledge cutoff), dẫn đến việc cung cấp thông tin lỗi thời.

RAG giải quyết các vấn đề này bằng cách chuyển hướng LLM truy xuất thông tin từ các nguồn kiến thức nội bộ có thẩm quyền và đã được xác thực trước khi sinh ra câu trả lời. Điều này mang lại cho tổ chức quyền kiểm soát tốt hơn đối với đầu ra của AI.

## 3. Lợi Ích Của Kiến Trúc RAG

- **Tiết kiệm chi phí phát triển:** Việc đào tạo lại (fine-tuning) một Mô hình nền tảng (Foundation Model - FM) cho một miền kiến thức cụ thể đòi hỏi chi phí tính toán và tài chính khổng lồ. RAG là phương pháp tiếp cận tiết kiệm chi phí hơn rất nhiều để bổ sung dữ liệu mới, giúp công nghệ AI tạo sinh dễ tiếp cận hơn.
- **Duy trì tính cập nhật:** RAG cho phép các nhà phát triển liên tục cung cấp nghiên cứu, tài liệu, hoặc tin tức mới nhất cho LLM. Hệ thống có thể kết nối với các nguồn dữ liệu thay đổi liên tục để mang lại thông tin theo thời gian thực.
- **Kiểm chứng và phân bổ nguồn (Source Attribution):** RAG cho phép LLM trình bày thông tin đi kèm với trích dẫn tài liệu nguồn. Người dùng có thể tự kiểm tra tài liệu gốc để làm rõ chi tiết, từ đó tăng độ tin cậy của hệ thống.
- **Bảo mật và Kiểm soát:** Các nhà phát triển có thể kiểm soát nguồn thông tin, hạn chế truy xuất các dữ liệu nhạy cảm dựa trên mức độ phân quyền, và dễ dàng khắc phục sự cố bằng cách chỉnh sửa tài liệu nguồn nếu LLM tham chiếu sai.

## 4. Các Công Cụ Truy Xuất (Retrievers)

Tùy thuộc vào nhu cầu tìm kiếm theo ngữ nghĩa hay từ khóa, hệ thống có thể sử dụng các loại cơ sở dữ liệu sau:

- **Vector Database (Cơ sở dữ liệu Vector):** Chuyển đổi câu truy vấn thành các vector dày đặc (dense vector) thông qua các mô hình nhúng (như BERT, RoBERTa, hoặc MiniLM). Việc tìm kiếm được thực hiện dựa trên độ tương đồng về ngữ nghĩa (semantic similarity) giữa vector truy vấn và vector tài liệu.
- **Graph Database (Cơ sở dữ liệu Đồ thị):** Xây dựng kho tri thức dựa trên mối quan hệ giữa các thực thể (entities) được trích xuất từ văn bản. Phương pháp này mang lại độ chính xác cao nhưng đòi hỏi cấu trúc truy vấn chặt chẽ.
- **Regular SQL Database:** Lưu trữ và truy xuất dữ liệu có cấu trúc truyền thống nhưng thiếu sự linh hoạt trong việc hiểu ngữ nghĩa tự nhiên.

## 5. Quy Trình Hoạt Động Của Hệ Thống RAG (Workflow)

Quy trình thực thi của hệ thống RAG trong dự án bao gồm 5 bước cốt lõi:

1.  **Create Vector Database (Tạo cơ sở dữ liệu):** Chuyển đổi toàn bộ tài liệu tri thức (.md, .txt) thành các vector thông qua Embedding Model và lưu trữ vào Vector DB (ChromaDB).
2.  **User Input (Người dùng truy vấn):** Người dùng cung cấp một câu truy vấn bằng ngôn ngữ tự nhiên (qua giọng nói hoặc văn bản) để tìm kiếm thông tin.
3.  **Information Retrieval (Truy xuất thông tin):** Cơ chế Retriever quét toàn bộ database để tính toán toán học, từ đó xác định các phân đoạn tri thức (chunks/paragraphs) có ngữ nghĩa tương đồng nhất với câu truy vấn.
4.  **Combining Data (Kết hợp dữ liệu):** Các đoạn văn bản truy xuất được sẽ được ghép nối với câu truy vấn ban đầu của người dùng để tạo thành một Prompt hoàn chỉnh (giàu ngữ cảnh).
5.  **Generate Text (Sinh phản hồi):** Prompt đã được làm giàu ngữ cảnh sẽ được đưa vào LLM để sinh ra câu trả lời cuối cùng, đảm bảo tính chính xác và bám sát tài liệu nội bộ.

## 2. Mục tiêu

Dự án này xây dựng một MCP server cung cấp khả năng tra cứu tri thức (RAG - Retrieval Augmented Generation) cho robot Xiaozhi. Khi người dùng hỏi robot một câu hỏi liên quan đến dự án (kiến trúc, cách hoạt động, quyết định kỹ thuật...), robot sẽ gọi tool `search_knowledge` để tìm đoạn thông tin liên quan nhất trong kho tri thức nội bộ, rồi dùng kết quả đó để trả lời bằng giọng nói.

## 3. Vì sao chọn đề tài RAG + MCP cho robot Xiaozhi

- **Tính thực tiễn:** Thay vì chỉ làm chatbot terminal/web, việc tích hợp AI vào một thiết bị phần cứng (robot ESP32) cho phép demo trực quan: hỏi bằng giọng nói, robot phản hồi bằng giọng nói, thể hiện khả năng ứng dụng AI vào thực tế (AIoT).
- **Bắt kịp công nghệ:** MCP là chuẩn giao tiếp mới giúp LLM gọi tool và truy cập dữ liệu ngoài một cách có cấu trúc, đang được nhiều hệ thống AI hiện nay áp dụng.
- **Kiến trúc tách rời (decoupled):** Thiết bị (robot) – bộ não (LLM ở backend) – nguồn tri thức (MCP server + RAG) là 3 lớp độc lập. Nếu RAG trả lời sai, chỉ cần sửa ở MCP server, không cần đụng đến firmware robot — giúp việc gỡ lỗi (debug) nhanh hơn nhiều.
- **Chạy được cục bộ trước khi triển khai:** Toàn bộ pipeline (embedding, vector DB, MCP server) có thể chạy và test trên máy cá nhân trước khi cân nhắc đưa lên server mạnh hơn.

## 4. Kiến trúc tổng thể

Luồng dữ liệu của hệ thống được mô tả như sau:

```text
[Người dùng nói]
       │
       ▼
[Robot Xiaozhi - ESP32 firmware]
       │  WebSocket / MQTT
       ▼
[Xiaozhi backend server]  (ASR → LLM → TTS, điều phối hội thoại)
       │  WebSocket (MCP_ENDPOINT, giao thức JSON-RPC 2.0)
       ▼
[MCP server của dự án này]  (chạy trên PC/server riêng, qua mcp_pipe.py)
       │
       ▼
[knowledge_server.py]  →  gọi tool search_knowledge(query)
       │
       ▼
[Chroma Vector Database]  (đã embed sẵn từ knowledge_base/)
       │
       ▼
[Trả về đoạn văn bản liên quan nhất] → LLM tổng hợp câu trả lời → TTS → Robot nói ra
```

Có 2 nhóm MCP khác nhau trong hệ sinh thái Xiaozhi, dự án này dùng nhóm thứ hai:

1. **MCP trên thiết bị** (device-side, trong firmware `xiaozhi-esp32`): điều khiển phần cứng
   robot trực tiếp (đèn, servo, âm lượng...). Không dùng trong phạm vi dự án này.
2. **MCP server ngoài** (cloud-side, giống mẫu `mcp-calculator`): một tiến trình Python độc lập,
   kết nối vào backend qua WebSocket bằng `mcp_pipe.py`, expose các tool tùy chỉnh cho AI model gọi.
   **Đây chính là mô hình dự án này sử dụng** để expose tool `search_knowledge`.

## 5. Pipeline nạp dữ liệu (Ingestion)

Được thực hiện qua 2 script chính:

### 5.1. Crawl dữ liệu thô

Tải các file `.md` liên quan từ các repo nguồn (README, tài liệu MCP protocol, mẫu MCP server...)
về thư mục `knowledge_base/` dưới dạng text thuần.

### 5.2. Nạp vào Vector Database (`ingest.py`)

| Bước          | Công cụ                                                                 | Vai trò                                                                                                  |
| ------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Đọc tài liệu  | `DirectoryLoader` + `TextLoader` (langchain_community)                  | Quét toàn bộ `.md` trong `knowledge_base/`                                                               |
| **Chunking**  | `RecursiveCharacterTextSplitter`                                        | Chia văn bản thành đoạn ~1000 ký tự, overlap 200, ưu tiên cắt theo heading (`##`, `###`) để giữ ngữ cảnh |
| **Embedding** | `HuggingFaceEmbeddings` (model `paraphrase-multilingual-MiniLM-L12-v2`) | Chuyển mỗi đoạn văn bản thành vector số học để so sánh ngữ nghĩa                                         |
| **Lưu trữ**   | `Chroma` (langchain_chroma)                                             | Lưu vector + văn bản gốc vào `./vector_db`, cho phép truy vấn similarity search                          |

**Vì sao chọn `all-MiniLM-L6-v2` cho embedding:**

- **Chạy tốt trên CPU:** model nhẹ, không cần GPU, phù hợp máy cá nhân khi phát triển và test.
- **Vector nhỏ (384 chiều):** giúp ChromaDB truy vấn nhanh và tốn ít dung lượng lưu trữ hơn so với
  các model embedding lớn (768–1536 chiều).
- **Miễn phí, chạy local:** không phụ thuộc API trả phí (OpenAI, v.v.), không lo hết quota khi test
  liên tục trong quá trình phát triển.
- **Đủ tốt cho tài liệu tiếng Anh/kỹ thuật:** phù hợp với phần lớn tài liệu nguồn (docs kỹ thuật của
  repo Xiaozhi) chủ yếu bằng tiếng Anh.
  **Sửa lỗi kỹ thuật quan trọng trong quá trình làm:** ban đầu dự án dùng `HuggingFaceBgeEmbeddings`
  với model `all-MiniLM-L6-v2`. Qua kiểm thử phát hiện đây là lựa chọn sai: `HuggingFaceBgeEmbeddings`
  là class dành riêng cho họ model BGE, mặc định tự thêm câu instruction
  _"Represent this question for searching relevant passages: "_ vào **mọi câu query**, nhưng
  **không** thêm gì vào văn bản tài liệu (document). Cách này chỉ đúng khi model được huấn luyện theo
  kiểu bất đối xứng (asymmetric) đó — tức đúng model BGE thật. Áp lên `all-MiniLM-L6-v2` (không cùng
  họ, không được train theo cách này) sẽ làm vector câu hỏi bị lệch so với vector tài liệu, giảm độ
  chính xác similarity search. Đã sửa bằng cách chuyển sang `HuggingFaceEmbeddings`
  (package `langchain_huggingface`) — đúng chuẩn, đối xứng, không tự ý chèn instruction lạ.

### Vì sao chọn `paraphrase-multilingual-MiniLM-L12-v2` cho embedding

Ban đầu dự án dùng `all-MiniLM-L6-v2` — nhẹ, nhanh, nhưng qua script test độ chính xác
(`test_accuracy.py`) phát hiện: model này chủ yếu huấn luyện trên tiếng Anh, có Recall@3 chỉ
37.5% khi truy vấn bằng tiếng Việt trên tài liệu tiếng Anh (README, docs kỹ thuật gốc) — vấn
đề cross-lingual retrieval.

Đã chuyển sang `paraphrase-multilingual-MiniLM-L12-v2` — cùng họ MiniLM (vẫn nhẹ, chạy tốt
CPU), nhưng hỗ trợ 50+ ngôn ngữ bao gồm tiếng Việt, giúp khớp ngữ nghĩa xuyên ngôn ngữ tốt
hơn hẳn. Sau khi đổi, các câu hỏi từng bị miss hoàn toàn (không nằm trong top-10) đã vào
được top-3 đến top-5.

**Vì sao chọn ChromaDB:** chạy được ở dạng thư viện nhúng (embedded), lưu trực tiếp vào thư mục dự
án (`./vector_db`), không cần dựng thêm hạ tầng (Docker, server riêng) như các vector DB dạng dịch
vụ (Milvus tự host, hoặc Pinecone — dịch vụ cloud có phí) — phù hợp cho giai đoạn phát triển và demo.

**Vì sao `chunk_size=1000`, `chunk_overlap=200`:** kích thước này đủ để một chunk chứa trọn một ý
hoàn chỉnh (ví dụ một đoạn hướng dẫn) mà không vượt quá context window của LLM khi tổng hợp câu trả
lời; overlap 200 giúp tránh cắt đứt ý ngay tại ranh giới giữa hai chunk liền kề.

**Không crawl toàn bộ ~90 board trong `xiaozhi-esp32/main/boards/`:** vì dữ liệu không liên quan đến
phạm vi dự án sẽ gây nhiễu ngữ nghĩa, làm retrieval kém chính xác hơn — chỉ nạp đúng phạm vi cần.

## 6. MCP Server phục vụ truy vấn (`knowledge_server.py`)

```python
from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

mcp = FastMCP("KnowledgeSearch")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./vector_db", embedding_function=embeddings)

@mcp.tool()
def search_knowledge(query: str) -> dict:
    """Tìm kiếm thông tin trong kho tri thức nội bộ về dự án RAG + MCP này."""
    docs = vectorstore.similarity_search(query, k=3)
    return {"success": True, "result": [d.page_content for d in docs]}

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Chạy server thông qua `mcp_pipe.py` (theo mẫu từ `mcp-calculator`), kết nối vào backend Xiaozhi
bằng biến môi trường `MCP_ENDPOINT`:

```bash
export MCP_ENDPOINT=<lấy từ cấu hình agent trên Xiaozhi>
python mcp_pipe.py knowledge_server.py
```

## 7. Luồng xử lý khi robot trả lời câu hỏi

1. Người dùng hỏi robot, ví dụ: _"Hệ thống RAG này hoạt động thế nào?"_
2. Backend Xiaozhi nhận diện cần dùng tool ngoài, gửi `tools/call` tới MCP server qua WebSocket.
3. `knowledge_server.py` nhận query, gọi `vectorstore.similarity_search()` để tìm 3 đoạn văn bản
   liên quan nhất trong `vector_db`.
4. Kết quả trả về cho LLM ở backend, LLM tổng hợp thành câu trả lời tự nhiên.
5. Câu trả lời được chuyển thành giọng nói (TTS) và robot phát ra.

## 8. Các thành phần chính của dự án

| Thành phần        | File                                    | Vai trò                                     |
| ----------------- | --------------------------------------- | ------------------------------------------- |
| Crawl dữ liệu     | `crawl.py`                              | Tải tài liệu nguồn từ GitHub                |
| Nạp vector DB     | `ingest.py`                             | Chunk + embed + lưu Chroma                  |
| MCP tool server   | `knowledge_server.py`                   | Expose tool `search_knowledge` cho AI model |
| Cầu nối WebSocket | `mcp_pipe.py` (từ mẫu `mcp-calculator`) | Kết nối MCP server với backend Xiaozhi      |

## 9. Hướng phát triển tiếp theo

- Bổ sung thêm tool khác ngoài `search_knowledge` (vd: `get_project_status`, `explain_component`)
  theo đúng pattern `@mcp.tool()` của FastMCP.
- Log lại các câu hỏi robot không tìm được câu trả lời tốt (similarity score thấp) để biết cần
  bổ sung tài liệu nào vào `knowledge_base/`.
- Cân nhắc thêm bước rerank sau `similarity_search` nếu số lượng tài liệu tăng lên, để cải thiện
  độ chính xác truy xuất.
