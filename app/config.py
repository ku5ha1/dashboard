from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    LLM_MODEL: str 
    DATABASE_URL: str
    DATA_CSV_PATH: str 

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()