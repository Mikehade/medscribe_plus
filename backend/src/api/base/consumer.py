"""
Base WebSocket consumer with authentication and connection management.
"""
import uuid
import asyncio
import json
import pytz
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

# from src.infrastructure.middleware.socket import WebSocketAuthMiddleware
# from src.infrastructure.db.models.auth import AuthProfile
from utils.logger import get_logger

logger = get_logger()


class BaseWebSocketConsumer(ABC):
    """
    Abstract base class for WebSocket consumers.
    
    Responsibilities:
    - Handle WebSocket connection lifecycle
    - Authenticate connections
    - Manage connection state
    - Define contract for message handling
    
    SOLID Principles:
    - Single Responsibility: Manages WebSocket lifecycle only
    - Open/Closed: Open for extension via abstract methods
    - Liskov Substitution: All implementations can be used interchangeably
    - Interface Segregation: Clear, focused interface
    - Dependency Inversion: Depends on abstractions (WebSocketAuthMiddleware)
    """
    
    def __init__(
        self,
        websocket: WebSocket,
        user: Any = None
    ) -> None:
        """
        Initialize the WebSocket consumer.
        
        Args:
            websocket: FastAPI WebSocket connection
            auth_middleware: Authentication middleware for validating connections
        """
        self.websocket = websocket
        # self.auth_middleware = auth_middleware
        # self.user: Optional[AuthProfile] = None
        self.user = None
        self.is_authenticated = False
    
    async def connect(self) -> bool:
        """
        Handle WebSocket connection and authentication.
        Template method that orchestrates the connection process.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Accept the WebSocket connection
            _ = await self.websocket.accept()
            logger.info("WebSocket connection accepted")
            
            # When decided, the authentication logic can be moved here

            _ = await self.on_connect()
            
            return True
            
        except Exception as e:
            logger.error(f"Error during WebSocket connection: {e}", exc_info=True)
            try:
                _ = await self.websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except:
                pass
            return False
    
    async def disconnect(self, code: int = status.WS_1000_NORMAL_CLOSURE) -> None:
        """
        Handle WebSocket disconnection and cleanup.
        
        Args:
            code: WebSocket close code
        """
        try:
            logger.info(f"Disconnecting WebSocket for user: {self.user.user_id if self.user else 'unknown'}")
            
            # Call subclass hook for pre-disconnection cleanup
            _ = await self.on_disconnect()
            
            # Close the WebSocket
            if not self.websocket.client_state.name == "DISCONNECTED":
                _ = await self.websocket.close(code=code)
                
        except Exception as e:
            logger.error(f"Error during WebSocket disconnection: {e}", exc_info=True)
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """
        Receive and parse JSON message from WebSocket.
        
        Returns:
            Parsed message dictionary or None if error
        """
        try:
            data = await self.websocket.receive_json()
            # logger.debug(f"Received message: {data}")
            return data
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected by client")
            raise
        except Exception as e:
            logger.error(f"Error receiving message: {e}", exc_info=True)
            return None
    
    async def _send_message(self, message: Dict[str, Any]) -> None:
        """
        Send JSON message through WebSocket.
        
        Args:
            message: Dictionary to send as JSON
        """
        try:
            _ = await self.websocket.send_json(message)
            # logger.debug(f"Sent message: {message}")
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
            # raise
    
    # Complete this
    async def send_error(self, message: str, code: int = 400) -> None:
        """
        Send error message through WebSocket.
        
        Args:
            message: Error message
            code: Error code
        """
        error_data = {
            # "type": "error",
            # "success": False,
            "data": message,
            "code": code,
            "event": "elle_bot.unknown_error",
            "error": "An unknown error occurred",
        }
        _ = await self._send_message(error_data)
    
    async def send_bot_message(
        self,
        text: str,
        sender: str,
        sender_full_name: str,
        event_name: str,
        encryption_key: str = None,
        status_details: Dict[str, bool] = {},
        other_details: Dict[str, Any] = {},
        echo: bool = False,
        upload: bool = False
    ) -> None:
        """
        Send message from bot to user
        
        Args:
            text: message text to send
            sender: sender of message
            sender_full_name: sender full name
            event_name: event name of the message
            encryption_key: encryption key for aes 256 encryption which will be activated
            status_details: - status details for UI which is a dict that varies
            other_details: Other details,
            echo: is this echoing back user message,
            upload: for file upload
        """
        message = {
            "sender": sender,
            "message_id": str(uuid.uuid4()),
            "sender_full_name": sender_full_name,
            "event": {
                "name": event_name,
                "status_details": status_details,
                "other_details": other_details,
                "encryption_key": encryption_key
            },
            "message": [{"text": text}] if not echo else {"text": text} if not upload else {"text": text, "fileId": "my file id", "fileName": "my file name"},
            "full_name": sender_full_name,
            "timestamp": datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
        }
        _ = await self._send_message(message)

    async def send_message_history(
        self,
        event_name: str,
        encryption_key: str = None,
        messages: list[Dict[str, Any]] = [],
        status_details: Dict[str, bool] = {},
        other_details: Dict[str, Any] = {},
    ) -> None:
        """
        Send message from bot to user
        
        Args:
            messages: formatted list of messages
            event_name: event name of the message
            encryption_key: encryption key for aes 256 encryption which will be activated
            status_details: - status details for UI which is a dict that varies
        """
        message = {
            "event": {
                "name": event_name,
                "status_details": status_details,
                "other_details": other_details,
                "encryption_key": encryption_key
            },
            "messages": messages,
        }
        _ = await self._send_message(message)
    
    async def handle_connection(self) -> None:
        """
        Main connection handler. Template method.
        Manages the entire WebSocket lifecycle.
        """
        connected = await self.connect()
        
        if not connected:
            return
        
        try:
            # Main message loop
            while True:
                message = await self.receive_message()
                
                if message is None:
                    continue
                
                # Route message to appropriate handler
                await self.handle_message(message)
                
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        finally:
            _ = await self.disconnect()


    async def heartbeat(self) -> None:
        """Send periodic heartbeat pings."""
        try:
            while self.connected:
                _ = await self._send_message({
                    "messages": {"text": "ping"},
                    "event": {"name": "heartbeat"}
                })
                _ = await asyncio.sleep(40)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat error: {e}", exc_info=True)

    
    # --- Abstract Methods (must be implemented by subclasses) ---
    @abstractmethod
    async def on_connect(self):
        """
        Hook called after successful connection and authentication.
        Subclasses should implement initialization logic here.
        """
        pass
    
    @abstractmethod
    async def on_disconnect(self):
        """
        Hook called before disconnection.
        Subclasses should implement cleanup logic here.
        """
        pass
    
    @abstractmethod
    async def handle_message(self, message: Dict[str, Any]):
        """
        Handle incoming WebSocket message.
        Subclasses must implement their message routing logic here.
        
        Args:
            message: Parsed message dictionary
        """
        pass