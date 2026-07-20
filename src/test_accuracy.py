import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR,"..", "vector_db")
K = 5
 
TEST_SET = [
    {
        "query": "RAG là gì",
        "expected_keyword": "Retrieval-Augmented Generation",
        "note": "Định nghĩa RAG trong project_overview.md mục 1",
    },
    {
        "query": "Vì sao chọn model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 cho embedding",
        "expected_keyword": "cross-lingual",
        "note": "Lý do chọn model, mục 5.2 project_overview.md",
    },
    {
        "query": "Lỗi HuggingFaceEmbeddings là gì",
        "expected_keyword": "MiniLM",
        "note": "Code mẫu MCP server, mục 6 project_overview.md",
    },
    {
        "query": "MCP tool search_knowledge trả về gì",
        "expected_keyword": "FastMCP",
        "note": "Code mẫu MCP server, mục 6 project_overview.md",
    },
    {
        "query": "Xiaozhi hỗ trợ bao nhiêu loại board phần cứng",
        "expected_keyword": "70+",
        "note": "README.md gốc từ xiaozhi-esp32",
    },
    {
        "query": "Thiết bị gửi tin nhắn hello khi mở kết nối WebSocket như thế nào",
        "expected_keyword": "hello",
        "note": "docs_websocket.md, phần mô tả handshake",
    },
    {
        "query": "Tại sao không nên ghi đè cấu hình board có sẵn",
        "expected_keyword": "OTA",
        "note": "docs_custom-board.md, cảnh báo về OTA update",
    },
    {
        "query": "Partition v2 có gì khác so với v1",
        "expected_keyword": "assets",
        "note": "partitions_v2_README.md",
    },
]
 
 
 
def load_vectorstore():
    if not os.path.isdir(VECTOR_DB_DIR):
        raise FileNotFoundError(
            f"Không tìm thấy vector_db tại '{VECTOR_DB_DIR}'"
        )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
 
 
def run_accuracy_test(vectorstore, test_set, k=K, verbose=True):
    results = []
    for case in test_set:
        docs = vectorstore.similarity_search(case["query"], k=k)
        hit = any(case["expected_keyword"].lower() in d.page_content.lower() for d in docs)
        results.append({**case, "hit": hit, "retrieved": [d.page_content[:80] for d in docs]})
 
        if verbose:
            status = "✅ HIT " if hit else "❌ MISS"
            print(f"{status} | {case['query']}")
            if not hit:
                print(f" (Cơ sở ban đầu '{case['expected_keyword']}' — {case['note']})")
                for i, snippet in enumerate(results[-1]["retrieved"]):
                    print(f" top-{i+1}: {snippet}...")
 
    total = len(results)
    hits = sum(r["hit"] for r in results)
    accuracy = hits / total * 100 if total else 0
 
    print(f"KẾT QUẢ: {hits}/{total} câu đúng — Recall@{k} = {accuracy:.1f}%")
    return results, accuracy
 
 
def find_actual_rank(vectorstore, query, expected_keyword, max_k=10):
    """Tìm xem đoạn chứa từ khóa đúng thực ra nằm ở hạng (rank) thứ mấy,
    thay vì chỉ biết có/không nằm trong top-k cố định. Giúp phân biệt
    'suýt trúng' (rank 4-5) với 'hoàn toàn lạc đề' (không có trong top-10)."""
    docs = vectorstore.similarity_search(query, k=max_k)
    for rank, doc in enumerate(docs, start=1):
        if expected_keyword.lower() in doc.page_content.lower():
            return rank
    return None 
 
 
def diagnose_misses(vectorstore, results):
    for r in results:
        if not r["hit"]:
            rank = find_actual_rank(vectorstore, r["query"], r["expected_keyword"], max_k=10)
            if rank:
                print(f"🟡 '{r['query']}' -> đoạn đúng nằm ở HẠNG {rank} (ngoài top-{K}, nhưng có trong top-10)")
            else:
                print(f"🔴 '{r['query']}' -> đoạn đúng KHÔNG nằm trong top-10 luôn — vấn đề nặng hơn")
 
 
if __name__ == "__main__":
    vectorstore = load_vectorstore()
    print(f"Tổng số vector trong DB: {vectorstore._collection.count()}\n")
    results, accuracy = run_accuracy_test(vectorstore, TEST_SET)
    diagnose_misses(vectorstore, results)
 