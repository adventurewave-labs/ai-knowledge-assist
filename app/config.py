from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    CORS_ORIGINS: list[str] = ["*"]
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "aria-rag"
    LANGSMITH_TRACING: bool = False
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-4o-mini"
    FALLBACK_LLM_MODEL: str = "gemini-1.5-flash"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K: int = 5
    LOG_LEVEL: str = "INFO"

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}")
        return upper


settings = Settings()
