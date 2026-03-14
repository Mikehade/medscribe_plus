from typing import Optional
from typing import List, Optional
from pydantic import Field
from src.config.base import Settings

class StagingSettings(Settings):
    LOG_LEVEL: str = "DEBUG"

    BASE_URL: str = "http://localhost:8000"
    PORT: str = "8000"

    CORS_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS: list[str] = ["Authorization", "Content-Type"]
    CORS_ORIGINS: List[str] = Field(default=["*"]) 