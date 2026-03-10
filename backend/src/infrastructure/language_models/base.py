"""
Base abstract class for LLM providers.
Defines the common interface that all LLM providers must implement.
"""
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from jinja2 import Template

from utils.logger import get_logger

logger = get_logger()


class BaseLLMModel(ABC):
    """
    Abstract base class for all LLM providers.
    
    Provides common functionality like template rendering and defines
    the interface that all concrete implementations must follow.
    """
    
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.85,
        max_tokens: int = 4096,
        timezone_str: str = "UTC",
        **kwargs
    ):
        """
        Initialize base LLM model.
        
        Args:
            model_id: The model identifier
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timezone_str: Timezone for datetime rendering
            **kwargs: Additional provider-specific parameters
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timezone_str = timezone_str
        self.tool_registry = kwargs.get("tool_registry", None)
    
    @abstractmethod
    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        stream: bool = False,
        enable_tools: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke the LLM model with messages.
        
        Args:
            messages: List of message dictionaries with role and content
            system: Optional system prompt
            stream: Whether to stream the response
            enable_tools: Whether to enable tool/function calling
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Response dictionary from the model
        """
        pass
    
    @abstractmethod
    async def prompt(
        self,
        text: Optional[str] = None,
        system_prompt: str = "",
        system_context: Optional[Dict[str, Any]] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        enable_tools: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        High-level prompt method with template rendering.
        
        Args:
            text: User message text
            system_prompt: System prompt template (supports Jinja2)
            system_context: Context variables for system prompt rendering
            message_history: Previous conversation messages
            stream: Whether to stream the response
            enable_tools: Whether to enable tool/function calling
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Response dictionaries from the model
        """
        pass
    
    @abstractmethod
    async def handle_tool_calls(
        self,
        response: Dict[str, Any],
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle tool calls in model response and continue conversation.
        
        Args:
            response: The model response containing tool calls
            messages: Current message history
            **kwargs: Additional parameters
            
        Returns:
            Final model response after tool execution
        """
        pass
    
    def render_template(
        self,
        template_string: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render Jinja2 template with context and datetime info.
        
        Args:
            template_string: Jinja2 template string
            context: Variables to render in template
            
        Returns:
            Rendered template string
        """
        now = datetime.now(ZoneInfo(self.timezone_str))
        default_context = {
            "datetime": now.strftime("%Y-%m-%d %H:%M"),
            "day_name": now.strftime("%A"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
        }
        
        if context:
            default_context.update(context)
        
        template = Template(template_string)
        return template.render(**default_context)

    def extract_text_from_llm_response(
        self, 
        content: Any, 
        include_tokens: bool = False, 
        llm_type: str = "bedrock"
    ) -> str:
        """
        Extract text from various content formats.
        
        Args:
            content: Can be:
                - List of dicts: [{"text": "..."}]
                - String: "..."
                - Dict: {"text": "..."}

            llm_type: - llm model type being in use
        
        Returns:
            Extracted text string
        """

        if llm_type.lower() == "bedrock":
            tokens_data = {
                # "input_tokens": content.get("usage", {}).get("inputTokens"),
                # "output_tokens": content.get("usage", {}).get("outputTokens"),
                "total_tokens": content.get("usage", {}).get("totalTokens"),
                "tool_tokens": content.get("toolTokens", 0),
                "tool_count": content.get("toolCallCount", 0)
            }
            logger.info(f"Tokens Data: {tokens_data}\n")
            # return content.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip()
            extracted = content.get("output", {}).get("message", {}).get("content", [{}])
            if len(extracted) > 0:
                return extracted[0].get("text", "").strip() if not include_tokens else (extracted[0].get("text", "").strip(),
                                                                tokens_data)
            return "" if not include_tokens else ("", tokens_data)

        else:
            if isinstance(content, str):
                return content if not include_tokens else (content, {})
        
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                return " ".join(text_parts) if not include_tokens else (" ".join(text_parts), {})
            
            if isinstance(content, dict) and "text" in content:
                return content["text"] if not include_tokens else (content["text"], {})
            
            return str(content) if not include_tokens else (str(content), {})
    
    def format_tool_response(
        self,
        original_message: str,
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """
        Format tool execution results for model consumption.
        
        Args:
            original_message: The original user message
            tool_results: List of tool execution results
            
        Returns:
            Formatted string combining original message and results
        """
        results_text = f"User message: {original_message}\n\n"
        results_text += "Function call results:\n"
        
        for i, result in enumerate(tool_results, 1):
            tool_name = result.get("tool_name", "unknown")
            tool_output = result.get("output", "")
            results_text += f"\n{i}. {tool_name}:\n{tool_output}\n"
        
        return results_text


    @staticmethod
    def clean_message_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean message history to ensure valid user-assistant pairs.
        
        Filters out:
        - Trivial assistant responses ("success", very short replies)
        - Duplicate content
        - Invalid pairs
        
        Args:
            messages: Raw message history in format:
                [{"role": "user/assistant", "content": [{"text": "..."}]}]
        
        Returns:
            Cleaned message history
        """
        if not messages:
            return []
        
        cleaned = []
        i = 0
        
        while i < len(messages) - 1:
            current = messages[i]
            next_msg = messages[i + 1]
            
            # Look for valid user-assistant pairs
            if current["role"] == "user" and next_msg["role"] == "assistant":
                # Extract text content
                user_text = BaseLLMModel._extract_text_from_content(current["content"])
                assistant_text = BaseLLMModel._extract_text_from_content(next_msg["content"])
                
                # Clean assistant text
                assistant_text = assistant_text.replace("None", "").replace("null", "").strip()
                
                # Validate assistant response
                if BaseLLMModel._is_valid_assistant_response(assistant_text, user_text):
                    cleaned.append(current)
                    cleaned.append(next_msg)
                
                i += 2  # Move to next pair
            else:
                # Skip standalone messages
                i += 1
        
        # Ensure last message is from user (models expect this)
        # Remove trailing user message if no assistant response follows
        if cleaned and cleaned[-1]["role"] == "user":
            cleaned.pop()
        
        return cleaned
    
    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        """
        Extract text from various content formats.
        
        Args:
            content: Can be:
                - List of dicts: [{"text": "..."}]
                - String: "..."
                - Dict: {"text": "..."}

            llm_type: - llm model type being in use
        
        Returns:
            Extracted text string
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts)
        
        if isinstance(content, dict) and "text" in content:
            return content["text"]
        
        return str(content)
    
    @staticmethod
    def _is_valid_assistant_response(assistant_text: str, user_text: str) -> bool:
        """
        Check if assistant response is valid and meaningful.
        
        Args:
            assistant_text: Assistant's response
            user_text: User's message
        
        Returns:
            True if response is valid
        """
        if not assistant_text:
            return False
        
        # Filter out trivial responses
        if assistant_text.lower() in ["success", "ok", "done"]:
            return False
        
        # Ensure minimum length (at least 4 words)
        if len(assistant_text.split()) < 3:
            return False
        
        # Exclude if identical to user message
        if assistant_text == user_text:
            return False
        
        return True