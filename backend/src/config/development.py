from typing import Optional
from typing import List, Optional
from pydantic import Field
from src.config.base import Settings

class DevSettings(Settings):
    LOG_LEVEL: str = "DEBUG"

    BASE_URL: str = "http://localhost:8020"
    PORT: str = "8020"

    CORS_ORIGINS: List[str] = Field(default=["*"])  # Accept all by default
    CORS_METHODS: List[str] = Field(default=["*"])
    CORS_HEADERS: List[str] = Field(default=["*"])
