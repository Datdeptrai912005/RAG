import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
 
KNOWLEDGE_BASE_DIR = "./knowledge_base"
VECTOR_DB_DIR = "./vector_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
 

def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)          
    text = re.sub(r"\n\s*\n+", "\n\n", text)      
    return text.strip()
 
def main():
    print("Đang quét thư mục knowledge_base...")
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục '{KNOWLEDGE_BASE_DIR}'. "
            f"Hãy chạy script crawl dữ liệu trước."
        )
 
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        use_multithreading=True,
        show_progress=True,
    )
    documents = loader.load()
 
    if not documents:
        print("-> Không tìm thấy tài liệu nào. Dừng chương trình.")
        return
    print(f"-> Đã tìm thấy {len(documents)} tệp tài liệu.")
    
    for doc in documents:
        doc.page_content = clean_html(doc.page_content)

    print("\n Thực hiện giai đoạn chunking ")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"-> Đã chia thành {len(chunks)} đoạn nhỏ (chunks).")
 
    # 3. Tạo embeddings
    print(f"\n Đang tạo Embeddings bằng model {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("\nĐang lưu vào Vector DB (Chroma)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )
 
    print(f"\n-> NẠP DỮ LIỆU THÀNH CÔNG! {vectorstore._collection.count()} vectors đã được lưu tại '{VECTOR_DB_DIR}'.")
 
 
if __name__ == "__main__":
    main()