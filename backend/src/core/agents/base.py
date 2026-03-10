"""
Base Agent.
"""
import json
import uuid
import asyncio
from typing import Any, Dict, List, Optional, AsyncGenerator
from abc import abstractmethod, ABC
from datetime import datetime
from utils.logger import get_logger
# from src.infrastructure.db.models.auth import AuthProfile

logger = get_logger()

class BaseAgent(ABC):
    """
    Base agent class to be inherited by other agents
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    async def process_message(
        self,
        # user: AuthProfile,
        message: str,
        bot: str = "scribe",
        use_history: bool = True,
        enable_tools: bool = True,
        stream: bool = False
    ):
        """
        Process user message and generate response.
        Subclasses must implement their processing message logic here since each agent would 
        have the way their messages are processed.
        
        Args:
            bot: bot being used
            user: Authenticated user
            message: User's message text
            use_history: Determine if to use previous conversation messages for context
            enable_tools: Whether to enable tool calling
            stream: Whether to stream the response
        """
        pass

    async def ensure_async_generator(
        self, 
        result
    ) -> AsyncGenerator:
        """
        Ensures result is always an async generator.
        Forces async generators to complete execution before yielding.
        """
        # If already an async generator, consume it and re-yield
        if hasattr(result, "__aiter__"):
            async def materialized_generator():
                # Force the generator to execute completely
                items = []
                async for item in result:
                    items.append(item)
                
                # Now yield the collected items
                for item in items:
                    yield item
            
            return materialized_generator()
        
        # If it's a coroutine, await it first
        if asyncio.iscoroutine(result):
            result = await result
            # After awaiting, check again if it's an async generator
            if hasattr(result, "__aiter__"):
                async def materialized_generator():
                    items = []
                    async for item in result:
                        items.append(item)
                    for item in items:
                        yield item
                return materialized_generator()
        
        # If it's a normal iterable (list, tuple, generator)
        if hasattr(result, "__iter__"):
            async def async_gen():
                for item in result:
                    yield item
            return async_gen()
        
        raise TypeError(f"LLM prompt returned unsupported type: {type(result)}")