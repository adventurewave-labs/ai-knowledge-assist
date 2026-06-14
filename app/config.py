from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    OPENAI_API_KEY: str = "sk-placeholder"
    GOOGLE_API_KEY: str = ""
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

    # Comma-separated list of allowed CORS origins. Empty disables credentialed
    # cross-origin access and falls back to a wildcard without credentials.
    CORS_ORIGINS: str = ""

    # API key required on protected endpoints via the X-API-Key header. When
    # empty, authentication is disabled (suitable for local development/tests).
    API_KEY: str = ""

    # Path to the JSON file used to persist ingest deduplication keys across
    # restarts. Set to an empty string to keep dedup state in memory only.
    DEDUP_STORE_PATH: str = "./aria_dedup.json"

    # Log level for the structured JSON logger configured at startup.
    LOG_LEVEL: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS into a clean list of origins."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
