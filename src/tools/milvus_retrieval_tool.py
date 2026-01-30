from langchain.tools import tool
from langchain_core.tools import retriever

_db_instance = None

def get_milvus_db():
    global _db_instance
    if _db_instance is None:
        from src.vector_database.milvus_database import MilvusDB
        _db_instance = MilvusDB()
    return _db_instance

@tool
def milvus_retrieval(query: str):
    "This tool is used to extract relevant scientific papers"
    # Call the function with () to get the instance, then access the retriever
    db = get_milvus_db() 
    docs = db.retriever.invoke(query)
    
    formatted_docs = []
    for doc in docs:
        info = (
            f"CONTENT: {doc.page_content}\n"
            f"METADATA - Title: {doc.metadata.get('title', 'N/A')}, "
            f"Authors: {doc.metadata.get('authors', 'N/A')}, "
            f"DOI: {doc.metadata.get('doi', 'N/A')}"
        )
        formatted_docs.append(info)
    
    return "\n\n---\n\n".join(formatted_docs)