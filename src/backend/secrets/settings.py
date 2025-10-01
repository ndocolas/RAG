"""Application settings loaded from environment variables or a .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import find_dotenv
from pydantic import SecretStr


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=find_dotenv(), env_file_encoding="utf-8", extra="ignore"
    )

    GEMINI_API_KEY: SecretStr
    GEMINI_MODEL: str = "gemini-2.5-pro"
    EMBEDDING_MODEL: str = "model/embedding-001"
    EMBEDDING_DIM: int = 768

    TOP_K: int = 5

    LOG_LEVEL: str = "INFO"
    API_BASE_URL: str


settings = Settings()  # type: ignore
