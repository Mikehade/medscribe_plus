from typing import Optional
from typing import List, Optional
from pydantic import Field
from src.config.base import Settings

class ProductionSettings(Settings):
    LOG_LEVEL: str = "DEBUG"
    BASE_URL: str = "http://localhost:8020"
    PORT: str = "8020"

    CORS_METHODS: list[str] = ["GET", "POST"]
    CORS_HEADERS: list[str] = ["Authorization", "Content-Type"]