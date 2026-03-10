from typing import Any, Optional
from uuid import UUID
import os
import json
from typing import Literal
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, status, Depends, Query, File, UploadFile, HTTPException, Form
from src.config.dependency_injection.container import Container
# from src.infrastructure.db.models.auth import AuthProfile
# from src.infrastructure.middleware.dependencies import get_current_user
# from src.api.base.schemas import ( 
    
#                         )

from utils.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/Core", tags=["Core"])

@router.get("/tests")
async def test_endpoint():
    return {"message": "MedScribe API test"}