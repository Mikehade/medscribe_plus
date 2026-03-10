import os
from pathlib import Path
from pydantic import Field
from typing import Optional, List, Dict
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # parent of current directory
ENV_FILE = BASE_DIR / ".env"

# Get the current directory
current_directory = os.getcwd()
# Get the parent directory
parent_directory = os.path.dirname(current_directory)

class Settings(BaseSettings):
    APP_NAME: str = "Medscribe Agent API"
    DESCRIPTION: str = "MedScribe Agent API"
    API_ROOT_PATH: Optional[str] = Field(default="", validation_alias="API_ROOT_PATH")
    
    # Use validation_alias instead of env
    env: str = Field(..., validation_alias="APP_ENV")
    # JWT_SECRET: str = Field(..., validation_alias="JWT_SECRET")
    DEBUG: bool = True
    LOG_LEVEL: str = Field(..., validation_alias="LOG_LEVEL")
    # BASE_URL: str = "http://localhost:8007"
    # PORT: str = "8007"

    # PG_PORT: Optional[str] = Field(..., validation_alias="PG_PORT")
    # PG_DB_NAME: Optional[str] = Field(..., validation_alias="PG_DB_NAME")
    # PG_PASSWORD: Optional[str] = Field(..., validation_alias="PG_PASSWORD")
    # PG_HOSTNAME: Optional[str] = Field(..., validation_alias="PG_HOSTNAME")
    # PG_USERNAME: Optional[str] = Field(..., validation_alias="PG_USERNAME")
    AWS_ACCESS_KEY: Optional[str] = Field(..., validation_alias="AWS_ACCESS_KEY")
    AWS_SECRET_KEY: Optional[str] = Field(..., validation_alias="AWS_SECRET_KEY")
    AWS_REGION_NAME: Optional[str] = Field(..., validation_alias="AWS_REGION_NAME")
    NOVA_ACT_API_KEY: Optional[str] = Field(..., validation_alias="NOVA_ACT_API_KEY")
    # AWS_BUCKET_NAME: Optional[str] = Field(..., validation_alias="AWS_BUCKET_NAME")
    SERP_API_KEY: Optional[str] = Field(..., validation_alias="SERP_API_KEY")

    # Redis
    REDIS_DB: Optional[str] = Field(..., validation_alias="REDIS_DB")
    REDIS_PORT: Optional[str] = Field(..., validation_alias="REDIS_PORT")
    REDIS_NAME: Optional[str] = Field(..., validation_alias="REDIS_NAME")
    REDIS_PASSWORD: Optional[str] = Field(..., validation_alias="REDIS_PASSWORD")
    REDIS_HOST: Optional[str] = Field(..., validation_alias="REDIS_HOST")
    REDIS_LOCATION: Optional[str] = Field(..., validation_alias="REDIS_LOCATION")
    REDIS_URL: Optional[str] = Field(..., validation_alias="REDIS_URL")


    API_V_STR: str = "/api/v1"
    VERSION: str = "1.0.0"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return (
            f"postgresql://{self.PG_USERNAME}:{self.PG_PASSWORD}"
            f"@{self.PG_HOSTNAME}:{self.PG_PORT}/{self.PG_DB_NAME}"
        )

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        # env_file=".env",
        populate_by_name=True,
        extra="allow",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()