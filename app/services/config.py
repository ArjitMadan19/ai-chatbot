import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv


load_dotenv()


def get_int_env(name, default):
    value = os.getenv(name)

    if value is None:
        return default

    return int(value)


def get_bool_env(name, default):
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in {"1", "true", "yes", "on"}


def get_list_env(name, default):
    value = os.getenv(name)

    if value is None:
        return default

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def normalize_database_url(database_url):
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)

    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    return database_url


@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv("APP_TITLE", "RAG Chatbot API")
    app_description: str = os.getenv(
        "APP_DESCRIPTION",
        "API for asking questions across contracts, research papers, and notes."
    )
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    cors_origins: List[str] = field(
        default_factory=lambda: get_list_env(
            "CORS_ORIGINS",
            ["http://localhost:3000", "http://127.0.0.1:3000"]
        )
    )
    rate_limit: str = os.getenv("RATE_LIMIT", "10/minute")
    api_memory_turns: int = get_int_env("API_MEMORY_TURNS", 3)
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_enabled: bool = get_bool_env("CACHE_ENABLED", True)
    cache_ttl_seconds: int = get_int_env("CACHE_TTL_SECONDS", 300)

    database_url: str = normalize_database_url(
        os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/ai_chatbot"
        )
    )

    docs_dir: Path = Path(os.getenv("DOCS_DIR", "docs"))
    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", "vectorstore")

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model_name: str = os.getenv(
        "RERANKER_MODEL_NAME",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
    llm_max_new_tokens: int = get_int_env("LLM_MAX_NEW_TOKENS", 600)

    chunk_size: int = get_int_env("CHUNK_SIZE", 400)
    chunk_overlap: int = get_int_env("CHUNK_OVERLAP", 50)

    retrieval_k: int = get_int_env("RETRIEVAL_K", 12)
    retrieval_fetch_k: int = get_int_env("RETRIEVAL_FETCH_K", 20)
    rerank_top_n: int = get_int_env("RERANK_TOP_N", 3)

    allow_dangerous_deserialization: bool = get_bool_env(
        "ALLOW_DANGEROUS_DESERIALIZATION",
        True
    )

    allowed_upload_extensions: Set[str] = field(
        default_factory=lambda: {".txt", ".pdf"}
    )
    doc_type_folders: Dict[str, str] = field(
        default_factory=lambda: {
            "contract": "contracts",
            "research_paper": "research_papers",
            "notes": "notes"
        }
    )


settings = Settings()
