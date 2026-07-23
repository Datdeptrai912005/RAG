import os
import sys
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
 
load_dotenv()
 
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "Không tìm thấy GOOGLE_API_KEY. Hãy tạo file .env ở thư mục gốc project "
        "với nội dung: GOOGLE_API_KEY=<key_của_bạn> "
        "(lấy key tại https://aistudio.google.com/app/apikey)"
    )
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "..", "vector_db")
 
if not os.path.isdir(VECTOR_DB_DIR):
    raise FileNotFoundError(
        f"Không tìm thấy vector_db tại '{VECTOR_DB_DIR}'. "
        f"Hãy chạy ingest.py trước để tạo vector database."
    )
 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
 
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
 
template = """Bạn là một trợ lý AI chuyên nghiệp cho dự án Xiaozhi Robot.
Dựa vào ngữ cảnh được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng.
Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói không biết, đừng tự bịa đặt.
 
Ngữ cảnh:
{context}
 
Câu hỏi: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
 
 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
 
 
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
 
 
def debug_retrieval(query: str):
    docs = retriever.invoke(query)
    print(f"\nĐÃ TÌM THẤY {len(docs)} ĐOẠN DỮ LIỆU ")
    for i, doc in enumerate(docs):
        print(f"Đoạn {i + 1}: {doc.page_content[:100]}...")
 
EVAL_QUESTIONS = [
    "RAG là gì",
    "Vì sao chọn model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 cho embedding",
    "Lỗi HuggingFaceBgeEmbeddings là gì",
    "MCP tool search_knowledge trả về gì",
    "Xiaozhi hỗ trợ bao nhiêu loại board phần cứng",
    "Thiết bị gửi tin nhắn hello khi mở kết nối WebSocket như thế nào",
    "Tại sao không nên ghi đè cấu hình board có sẵn",
    "Partition v2 có gì khác so với v1",
]
 
OUT_OF_SCOPE_QUESTIONS = [
    "Giá bán robot Xiaozhi là bao nhiêu?",
    "Ai là CEO của công ty sản xuất Xiaozhi?",
    "Cách nấu phở bò truyền thống như thế nào?",
]
 
 
def run_batch_eval(questions, save_path=None, label="in-scope"):
    rows = []
    for i, q in enumerate(questions, start=1):
        print(f"\n[{label}] Câu {i}/{len(questions)}: {q}")
 
        start = time.time()
        try:
            docs = retriever.invoke(q)
        except Exception as e:
            docs = []
            print(f"[LỖI khi retrieval: {e}]")
        ends = time.time()
        retrieval_time = ends - start
 
        try:
            answer = chain.invoke(q)
        except Exception as e:
            answer = f"[LỖI khi gọi LLM: {e}]"
        t2 = time.time()
        generation_time = t2 - ends
        total_time = t2 - start
 
        print(answer)
        print(f"\n(retrieval: {retrieval_time:.2f}s | generation: {generation_time:.2f}s | tổng: {total_time:.2f}s)")
 
        rows.append({
            "question": q,
            "answer": answer,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
        })
 
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"# Kết quả đánh giá Generation — {label} (tuần 3)\n\n")
            f.write("| # | Câu hỏi | Câu trả lời | Retrieval | Generation  | Tổng | Đúng/Sai  | Ghi chú |\n")
            for i, r in enumerate(rows, start=1):
                answer_oneline = r["answer"].replace("\n", " ").replace("|", "\\|")
                f.write(
                    f"| {i} | {r['question']} | {answer_oneline} | "
                    f"{r['retrieval_time']:.2f} | {r['generation_time']:.2f} | {r['total_time']:.2f} | | |\n"
                )
        print(f"\n-> Đã lưu bảng kết quả tại: {save_path}")
 
    avg_total = sum(r["total_time"] for r in rows) / len(rows) if rows else 0
    print(f"\nThời gian trung bình mỗi câu: {avg_total:.2f}s")
 
    return rows
 
 
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Chạy: python generator.py --batch
        in_scope_path = os.path.join(BASE_DIR, "..", "eval_generation_report.md")
        run_batch_eval(EVAL_QUESTIONS, save_path=in_scope_path, label="in-scope")
 
        out_scope_path = os.path.join(BASE_DIR, "..", "eval_out_of_scope_report.md")
        run_batch_eval(OUT_OF_SCOPE_QUESTIONS, save_path=out_scope_path, label="out-of-scope")
    else:
        cauhoi = "Xiaozhi Robot là gì?"
        debug_retrieval(cauhoi)
        print("\n CÂU TRẢ LỜI TỪ AI ")
        response = chain.invoke(cauhoi)
        print(response)