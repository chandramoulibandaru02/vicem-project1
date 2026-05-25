from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    upload_dir: str = "app/uploads"
    model_name: str = "demo-model"
    model_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"
    model_base_url: str = ""
    model_timeout: int = 30
    model_temperature: float = 0.2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
