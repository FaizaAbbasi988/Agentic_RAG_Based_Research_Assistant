"""Configuration settings for Travel Planner"""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from pydantic import ConfigDict



class Settings(BaseSettings):
    tavily_api_key: str
    google_api_key:str
    milvus_api: str

    generation_model: str = "gemini-2.5-flash"

    embedding_model: str = "nomic-embed-text:latest"
    ollama_url: Optional[str] = "http://localhost:11434"

    milvus_uri: Optional[str]
    
    milvus_collection_name: str = "Scientific_RAG"
    milvus_vector_field: List[str] = ["dense", "sparse"]
    milvus_ranker_params: List[float] = [0.7, 0.3]
    milvus_search_kwargs: int = 3
    milvus_ranker_type: str = "weighted"
    milvus_consistency_level: str = "Bounded"

    retry_max_attempts: int = 3
    retry_initial_interval: int = 1

    LANGCHAIN_TRACING_V2:str
    LANGCHAIN_ENDPOINT:str
    LANGCHAIN_API_KEY:str
    LANGCHAIN_PROJECT:str
    HTTP_PROXY:str
    HTTPS_PROXY:str


    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  
    )

def get_settings() -> Settings:
    """Get cached settings instance"""
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = Settings()
    return get_settings._instance


settings = get_settings()