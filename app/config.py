from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown keys so optional services don't break
    )

    # Required core
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-5-mini-2025-08-07"
    DATABASE_URL: str = "sqlite:///./app.db"
    DATA_CSV_PATH: str = "app/data/education_dummy.csv"

    # Optional: translation and TTS (deferred)
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GOOGLE_PROJECT_ID: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    ELEVENLABS_VOICE_ID: Optional[str] = None

    # Optional: server/runtime
    UVICORN_HOST: str = "127.0.0.1"
    UVICORN_PORT: int = 8000
    LOG_LEVEL: str = "info"

    # Optional: Vercel Blob storage
    BLOB_READ_WRITE_TOKEN: Optional[str] = None

settings = Settings()