from langchain_milvus import Milvus, BM25BuiltInFunction
from src.config.settings import settings
from langchain_huggingface import HuggingFaceEmbeddings

URI = settings.milvus_uri
TOKEN = settings.milvus_api

class MilvusDB():
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = Milvus(
            embedding_function=embeddings,
            collection_name=settings.milvus_collection_name,
            builtin_function=BM25BuiltInFunction(),
            vector_field=settings.milvus_vector_field,
            connection_args={
                "uri": URI,
                "token": TOKEN,
                "secure": True,
                "async": False,
                "enable_prefetch_thread": False # 👈 ADD THIS
            },
            consistency_level="Bounded",
            drop_old=False,
            auto_id=True
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs = {"k": 3}, ranker_type="weighted", 
    ranker_params={"weights": [0.7, 0.3]})





