import os
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
 
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
    """Nối nội dung các đoạn tài liệu lại thành 1 chuỗi sạch, thay vì đưa nguyên
    list các Document object (kèm metadata thô) vào prompt."""
    return "\n\n".join(doc.page_content for doc in docs)
 
 
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
 
 
def debug_retrieval(query: str):
    """In ra các đoạn tài liệu tìm được, để kiểm tra retriever hoạt động đúng
    trước khi xem câu trả lời cuối cùng từ LLM."""
    docs = retriever.invoke(query)
    print(f"\nĐÃ TÌM THẤY {len(docs)} ĐOẠN DỮ LIỆU ")
    for i, doc in enumerate(docs):
        print(f"Đoạn {i + 1}: {doc.page_content[:100]}...")
 
 
if __name__ == "__main__":
    cauhoi = "Xiaozhi Robot là gì?"
 
    debug_retrieval(cauhoi)
 
    print("\n CÂU TRẢ LỜI TỪ AI ")
    response = chain.invoke(cauhoi)
    print(response)