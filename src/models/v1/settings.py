from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Set
from dotenv import find_dotenv


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=find_dotenv(), env_file_encoding="utf-8", extra="ignore")

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-pro"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 768
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "docs"
    MAX_FILE_MB: int = 15
    ALLOWED_EXTS: Set[str] = {"pdf", "txt"}
    CHUNK_TARGET_TOKENS: int = 800
    CHUNK_OVERLAP: float = 0.10
    TOP_K: int = 5
    MMR_LAMBDA: float = 0.5
    LOG_LEVEL: str = "INFO"


settings = Settings()  # type: ignore
