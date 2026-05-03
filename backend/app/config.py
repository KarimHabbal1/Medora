from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    database_url: str
    secret_key: str
    openai_api_key: Optional[str] = None
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    class Config:
        # Look for .env in both the CWD and backend/ subdirectory
        env_file = ("backend/.env", ".env")
        extra = "ignore"


settings = Settings()