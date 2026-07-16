from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./vector_db", embedding_function=embeddings)

cauhoi = "RAG là gì?"
docs = vectorstore.similarity_search(cauhoi, k=3)
for i, doc in enumerate(docs):
    print("Kết quả tìm kiếm dữ liệu {i+1}")
    print(doc.page_content)