from typing import Any, List, Optional, Dict, Literal
import json
import os 
from uuid import UUID
from fastapi import APIRouter, status, Depends, Query, File, UploadFile, HTTPException, Form
from utils.logger import get_logger
from src.api.soap.schemas import SoapRequest, SoapResponse
from src.infrastructure.services.soap import SOAPService
from src.config.dependency_injection.container import Container
from dependency_injector.wiring import Provide, inject
from fastapi.responses import JSONResponse

logger = get_logger()

router = APIRouter(prefix="/Soap", tags=["SOAP"])
@router.post("/soap", response_model=SoapResponse)
@inject
async def get_soap_note(
    request: SoapRequest,
    soap_service: SOAPService = Depends(Provide[Container.soap_service])
    ) -> JSONResponse:
    """ 
    Get soap note from cache.

    Args:
        request: SoapRequest object
        soap_service: SOAPService object
    

    response:
        JsonResponse: an object of soap data

    """   
    try: 
        soap_data = await soap_service.get_soap_note_from_cache(request.session_id)

        if not soap_data["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
            detail=soap_data["error"]
        )
    
        response = JSONResponse(content = {

            "status": "success",
            "data": soap_data.get("data", {})
        })
        
        return response
    
    except Exception as e:
        logger.error(f"Error getting soap note: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )