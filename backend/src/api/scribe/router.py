from typing import Any, Optional
import uuid
from uuid import UUID
import os
import json
from typing import Literal
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, status, Depends, Query, File, UploadFile, HTTPException, Form, WebSocket
from fastapi.responses import JSONResponse
from src.config.dependency_injection.container import Container
# from src.infrastructure.db.models.auth import AuthProfile
# from src.infrastructure.middleware.dependencies import get_current_user
from src.core.agents.scribe import ScribeAgent
from src.api.scribe.schemas import ( 
    ApproveRequest, ApproveResponse, AudioUploadResponse
)

from src.api.scribe.consumer import ScribeConsumer

from utils.logger import get_logger

logger = get_logger()

router = APIRouter(prefix="/Scribe", tags=["Scribe"])

@router.post("/upload", response_model=AudioUploadResponse)
@inject
async def upload_audio(
    file: UploadFile = File(...),
    patient_id: str = Form(default="P001"),
    session_id: Optional[str] = Form(default=None),
    agent: ScribeAgent = Depends(Provide[Container.scribe_agent]),
):
    """
    Option B: Upload a pre-recorded audio file.
    Returns full SOAP note + evaluation scores.
    """
    session_id = session_id or str(uuid.uuid4())

    try:
        # Save to temp file for Nova 2 Sonic processing
        suffix = os.path.splitext(file.filename)[1] or ".wav"

        session_id = None if session_id.lower() == "string" else session_id

        result = await agent.process_audio(
            audio=file,
            patient_id=patient_id,
            session_id=session_id,
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))

        return JSONResponse(content={
            "session_id": result["session_id"],
            "transcript": result.get("transcript", ""),
            "soap": result["soap"],
            "scores": result["scores"],
            "patient_context": result.get("patient_context", {}),
            "missing_fields": result.get("missing_fields", []),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approve", response_model=ApproveResponse)
@inject
async def approve_note(
    request: ApproveRequest,
    agent: ScribeAgent = Depends(Provide[Container.scribe_agent]),
):
    """
    Physician approval endpoint — inserts finalized note into EHR via Nova Act.
    """
    try:
        result = await agent.approve_and_insert(
            session_id=request.session_id,
            soap_data=request.soap,
            patient_id=request.patient_id,
        )
        return ApproveResponse(
            success=result.get("success", False),
            ehr_record_id=result.get("ehr_record_id"),
            message=result.get("message", ""),
        )
    except Exception as e:
        logger.error(f"Approval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_id}")
@inject
async def get_patient_context(
    patient_id: str, 
    agent: ScribeAgent = Depends(Provide[Container.scribe_agent]),
):
    """Get mock patient context for EHR panel."""
    result = await agent.patient_tools.execute_tool(
        "get_patient_history", {"patient_id": patient_id}
    )
    return JSONResponse(content=result)



#  Websocket endpoints
ws_router = APIRouter(prefix="/api", tags=["Scribe WebSocket"])

@ws_router.websocket("/transcribe/")
@inject
async def scribe_websocket(
    websocket: WebSocket,
    # ws_auth_middleware: WebSocketAuthMiddleware = Depends(Provide[Container.websocket_auth_middleware]),
    agent: ScribeAgent = Depends(Provide[Container.scribe_agent])
):
    """
    WebSocket endpoint for Elle Study Permit Application chat.
    
    Authentication is handled by the get_current_user_ws dependency.
    By the time this function executes, the user is already authenticated.
    
    Authentication methods:
        - Query parameter: ?auth_token=<jwt_token>
        - Authorization header: Bearer <jwt_token>
        - Sec-WebSocket-Protocol header: bearer, <jwt_token>
    
    Args:
        websocket: FastAPI WebSocket connection
        ws_auth_middleware: WebSocket authentication middleware
        agent: Study permit application agent
        message_repo: Message repository
        document_service: Document validation service
    """
    logger.info("In transcribe websocket")
    # user, error = await get_current_user_ws(websocket)

    # if not user:
    #     error_message = error.get("data", "Authentication failed") if error else "Authentication failed"
    #     error_code = error.get("status_code", 401) if error else 401
    #     # Raise WebSocketException which will be caught by FastAPI
    #     raise WebSocketException(
    #         code=status.WS_1008_POLICY_VIOLATION,
    #         reason=error_message
    #     )
    
    # logger.info(f"WebSocket connection for authenticated user: {user.user_id}")

    try:
        consumer = ScribeConsumer(
            websocket=websocket,
            # user=user,
            agent=agent,
        )
        # Handle the WebSocket connection lifecycle
        await consumer.handle_connection()
    except WebSocketDisconnect:
        logger.info("Client disconnected from scribe WebSocket")
    except Exception as e:
        logger.error(f"Error in scribe WebSocket: {e}", exc_info=True)