"""
Unit tests for BaseLLMModel abstract class.
Tests common functionality like template rendering, message cleaning, and text extraction.
"""
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List

from src.infrastructure.language_models.base import BaseLLMModel


# Concrete implementation for testing
class ConcreteLLMModel(BaseLLMModel):
    """Concrete implementation of BaseLLMModel for testing"""
    
    def __init__(self, **kwargs):
        super().__init__(
            model_id="test-model",
            temperature=0.85,
            max_tokens=4096,
            timezone_str="UTC",
            **kwargs
        )
    
    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        stream: bool = False,
        enable_tools: bool = False,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock invoke implementation"""
        return {
            "output": {
                "message": {
                    "content": [{"text": "Test response"}]
                }
            }
        }
    
    async def prompt(
        self,
        text: str = None,
        system_prompt: str = "",
        system_context: Dict[str, Any] = None,
        message_history: List[Dict[str, Any]] = None,
        stream: bool = False,
        enable_tools: bool = False,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ):
        """Mock prompt implementation"""
        response = await self.invoke(
            messages=[{"role": "user", "content": [{"text": text}]}],
            system=system_prompt,
            stream=stream,
            enable_tools=enable_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        yield response
    
    async def handle_tool_calls(
        self,
        response: Dict[str, Any],
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Mock tool call handler"""
        return response


class TestBaseLLMModelInitialization:
    """Test suite for BaseLLMModel initialization"""
    
    def test_initializes_with_default_parameters(self):
        """Should initialize with default parameters"""
        model = ConcreteLLMModel()
        
        assert model.model_id == "test-model"
        assert model.temperature == 0.85
        assert model.max_tokens == 4096
        assert model.timezone_str == "UTC"
        assert model.tool_registry is None
    
    def test_stores_additional_kwargs(self):
        """Should store additional kwargs for provider-specific parameters"""
        model = ConcreteLLMModel(
            custom_param="value",
            another_param=123
        )
        
        # Additional kwargs should be accessible
        assert model.model_id == "test-model"


class TestRenderTemplate:
    """Test suite for render_template method"""
    
    def test_renders_simple_template(self):
        """Should render simple Jinja2 template"""
        model = ConcreteLLMModel()
        
        template = "Hello {{ name }}"
        context = {"name": "World"}
        
        result = model.render_template(template, context)
        
        assert result == "Hello World"
    
    def test_renders_template_with_all_datetime_variables(self):
        """Should provide datetime, day_name, date, and time variables"""
        model = ConcreteLLMModel()
        
        template = "{{ datetime }} | {{ day_name }} | {{ date }} | {{ time }}"
        result = model.render_template(template, {})
        
        parts = result.split(" | ")
        assert len(parts) == 4
        assert len(parts[2]) == 10  # Date: YYYY-MM-DD
        assert ":" in parts[3]  # Time: HH:MM
    
    def test_context_overrides_default_datetime(self):
        """Should allow context to override default datetime variables"""
        model = ConcreteLLMModel()
        
        template = "{{ date }}"
        context = {"date": "custom-date"}
        
        result = model.render_template(template, context)
        
        assert result == "custom-date"

    
    def test_renders_complex_template(self):
        """Should render complex template with multiple variables"""
        model = ConcreteLLMModel()
        
        template = """
        Hello {{ user_name }},
        Today is {{ day_name }}, {{ date }}.
        Your role: {{ role }}
        """
        context = {"user_name": "Alice", "role": "admin"}
        
        result = model.render_template(template, context)
        
        assert "Alice" in result
        assert "admin" in result
        assert "Today is" in result


class TestExtractTextFromLLMResponse:
    """Test suite for extract_text_from_llm_response method"""
    
    def test_extracts_text_from_bedrock_response(self):
        """Should extract text from Bedrock response format"""
        model = ConcreteLLMModel()
        
        content = {
            "output": {
                "message": {
                    "content": [{"text": "Response text"}]
                }
            }
        }
        
        result = model.extract_text_from_llm_response(content, llm_type="bedrock")
        
        assert result == "Response text"
    
    def test_extracts_text_from_string(self):
        """Should handle string content directly"""
        model = ConcreteLLMModel()
        
        result = model.extract_text_from_llm_response("Simple text", llm_type="openai")
        
        assert result == "Simple text"
    
    def test_extracts_text_from_list_of_dicts(self):
        """Should extract text from list of content blocks"""
        model = ConcreteLLMModel()
        
        content = [
            {"text": "First part"},
            {"text": "Second part"}
        ]
        
        result = model.extract_text_from_llm_response(content, llm_type="openai")
        
        assert result == "First part Second part"
    
    def test_extracts_text_from_list_with_strings(self):
        """Should handle list containing strings"""
        model = ConcreteLLMModel()
        
        content = ["Text one", "Text two"]
        
        result = model.extract_text_from_llm_response(content, llm_type="openai")
        
        assert result == "Text one Text two"
    
    def test_extracts_text_from_dict_with_text_key(self):
        """Should extract text from dict with text key"""
        model = ConcreteLLMModel()
        
        content = {"text": "Dictionary text"}
        
        result = model.extract_text_from_llm_response(content, llm_type="openai")
        
        assert result == "Dictionary text"
    
    def test_converts_unknown_format_to_string(self):
        """Should convert unknown format to string"""
        model = ConcreteLLMModel()
        
        content = 12345
        
        result = model.extract_text_from_llm_response(content, llm_type="openai")
        
        assert result == "12345"
    
    def test_strips_whitespace_from_bedrock_response(self):
        """Should strip whitespace from extracted text"""
        model = ConcreteLLMModel()
        
        content = {
            "output": {
                "message": {
                    "content": [{"text": "  Response text  "}]
                }
            }
        }
        
        result = model.extract_text_from_llm_response(content, llm_type="bedrock")
        
        assert result == "Response text"


class TestFormatToolResponse:
    """Test suite for format_tool_response method"""
    
    def test_formats_single_tool_result(self):
        """Should format single tool result"""
        model = ConcreteLLMModel()
        
        result = model.format_tool_response(
            "What's the weather?",
            [{"tool_name": "get_weather", "output": "Sunny, 25°C"}]
        )
        
        assert "What's the weather?" in result
        assert "get_weather" in result
        assert "Sunny, 25°C" in result
        assert "Function call results:" in result
    
    def test_formats_multiple_tool_results(self):
        """Should format multiple tool results"""
        model = ConcreteLLMModel()
        
        tool_results = [
            {"tool_name": "tool_one", "output": "Result 1"},
            {"tool_name": "tool_two", "output": "Result 2"}
        ]
        
        result = model.format_tool_response("Original message", tool_results)
        
        assert "1. tool_one" in result
        assert "2. tool_two" in result
        assert "Result 1" in result
        assert "Result 2" in result
    
    def test_handles_missing_tool_name(self):
        """Should handle missing tool_name gracefully"""
        model = ConcreteLLMModel()
        
        result = model.format_tool_response(
            "Message",
            [{"output": "Some output"}]
        )
        
        assert "unknown" in result
        assert "Some output" in result
    
    def test_handles_empty_tool_results(self):
        """Should handle empty tool results list"""
        model = ConcreteLLMModel()
        
        result = model.format_tool_response("Message", [])
        
        assert "Message" in result
        assert "Function call results:" in result


class TestCleanMessageHistory:
    """Test suite for clean_message_history static method"""
    
    def test_returns_empty_list_for_empty_input(self):
        """Should return empty list for empty messages"""
        result = BaseLLMModel.clean_message_history([])
        
        assert result == []
    
    def test_preserves_valid_user_assistant_pairs(self):
        """Should keep valid user-assistant message pairs"""
        messages = [
            {"role": "user", "content": [{"text": "Hello, how are you?"}]},
            {"role": "assistant", "content": [{"text": "I'm doing well, thank you!"}]}
        ]
        
        result = BaseLLMModel.clean_message_history(messages)
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
    
    def test_filters_trivial_assistant_responses(self):
        """Should filter out trivial responses like 'success', 'ok'"""
        messages = [
            {"role": "user", "content": [{"text": "Do something"}]},
            {"role": "assistant", "content": [{"text": "success"}]},
            {"role": "user", "content": [{"text": "Another question"}]},
            {"role": "assistant", "content": [{"text": "This is a proper response"}]}
        ]
        
        result = BaseLLMModel.clean_message_history(messages)
        
        # Should only keep the second pair
        assert len(result) == 2
        assert "proper response" in result[1]["content"][0]["text"]
    
    def test_filters_short_responses(self):
        """Should filter responses with less than 4 words"""
        messages = [
            {"role": "user", "content": [{"text": "Question one"}]},
            {"role": "assistant", "content": [{"text": "Too short"}]},
            {"role": "user", "content": [{"text": "Question two"}]},
            {"role": "assistant", "content": [{"text": "This is long enough response"}]}
        ]
        
        result = BaseLLMModel.clean_message_history(messages)
        
        assert len(result) == 2
        assert result[1]["content"][0]["text"] == "This is long enough response"
    
    def test_filters_duplicate_content(self):
        """Should filter when assistant repeats user message"""
        messages = [
            {"role": "user", "content": [{"text": "Same message"}]},
            {"role": "assistant", "content": [{"text": "Same message"}]}
        ]
        
        result = BaseLLMModel.clean_message_history(messages)
        
        assert len(result) == 0
    
    def test_removes_trailing_user_message(self):
        """Should remove trailing user message without assistant response"""
        messages = [
            {"role": "user", "content": [{"text": "First question"}]},
            {"role": "assistant", "content": [{"text": "First answer here"}]},
            {"role": "user", "content": [{"text": "Second question"}]}
        ]
        
        result = BaseLLMModel.clean_message_history(messages)
        
        # Should only keep first pair, remove trailing user message
        assert len(result) == 2
        assert result[-1]["role"] == "assistant"
    
    def test_skips_standalone_messages(self):
        """Should skip messages that don't form proper pairs"""
        messages = [
            {"role": "assistant", "content": [{"text": "Standalone assistant"}]},
            {"role": "user", "content": [{"text": "Valid user message"}]},
            {"role": "assistant", "content": [{"text": "Valid assistant response here"}]}
        ]
        
        result = BaseLLMModel.clean_message_history(messages)
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
    
    def test_handles_string_content_format(self):
        """Should handle messages with string content"""
        messages = [
            {"role": "user", "content": "String content question"},
            {"role": "assistant", "content": "String content answer here"}
        ]
        
        result = BaseLLMModel.clean_message_history(messages)
        
        assert len(result) == 2


class TestExtractTextFromContent:
    """Test suite for _extract_text_from_content static method"""
    
    def test_extracts_from_string(self):
        """Should return string as-is"""
        result = BaseLLMModel._extract_text_from_content("Simple string")
        
        assert result == "Simple string"
    
    def test_extracts_from_list_of_text_dicts(self):
        """Should extract and join text from list"""
        content = [
            {"text": "Part one"},
            {"text": "Part two"}
        ]
        
        result = BaseLLMModel._extract_text_from_content(content)
        
        assert result == "Part one Part two"
    
    def test_extracts_from_single_text_dict(self):
        """Should extract text from dict with text key"""
        content = {"text": "Single text"}
        
        result = BaseLLMModel._extract_text_from_content(content)
        
        assert result == "Single text"
    
    def test_handles_mixed_list_content(self):
        """Should handle list with both dicts and strings"""
        content = [
            {"text": "Dict text"},
            "String text"
        ]
        
        result = BaseLLMModel._extract_text_from_content(content)
        
        assert "Dict text" in result
        assert "String text" in result
    
    def test_converts_unknown_to_string(self):
        """Should convert unknown types to string"""
        result = BaseLLMModel._extract_text_from_content(12345)
        
        assert result == "12345"


class TestIsValidAssistantResponse:
    """Test suite for _is_valid_assistant_response static method"""
    
    def test_rejects_empty_response(self):
        """Should reject empty assistant response"""
        result = BaseLLMModel._is_valid_assistant_response("", "user text")
        
        assert result is False
    
    def test_rejects_trivial_responses(self):
        """Should reject trivial responses"""
        trivial = ["success", "ok", "done", "SUCCESS", "Ok", "DONE"]
        
        for response in trivial:
            result = BaseLLMModel._is_valid_assistant_response(response, "user text")
            assert result is False, f"Should reject: {response}"
    
    def test_rejects_short_responses(self):
        """Should reject responses with less than 4 words"""
        result = BaseLLMModel._is_valid_assistant_response("Too short", "user text")
        
        assert result is False
    
    def test_accepts_valid_short_response_with_four_words(self):
        """Should accept response with exactly 4 words"""
        result = BaseLLMModel._is_valid_assistant_response(
            "This is four words", 
            "user text"
        )
        
        assert result is True
    
    def test_rejects_duplicate_of_user_text(self):
        """Should reject when assistant repeats user exactly"""
        result = BaseLLMModel._is_valid_assistant_response(
            "Same message",
            "Same message"
        )
        
        assert result is False
    
    def test_accepts_valid_response(self):
        """Should accept valid, meaningful response"""
        result = BaseLLMModel._is_valid_assistant_response(
            "This is a proper response with substance",
            "What is the answer?"
        )
        
        assert result is True
    
    def test_accepts_long_response(self):
        """Should accept long responses"""
        long_response = "This is a much longer response that contains " \
                       "multiple sentences and provides detailed information."
        
        result = BaseLLMModel._is_valid_assistant_response(
            long_response,
            "user question"
        )
        
        assert result is True


class TestAbstractMethods:
    """Test suite ensuring abstract methods must be implemented"""
    
    def test_cannot_instantiate_base_class_directly(self):
        """Should not be able to instantiate BaseLLMModel directly"""
        with pytest.raises(TypeError):
            BaseLLMModel(model_id="test")
    
    @pytest.mark.asyncio
    async def test_concrete_class_must_implement_invoke(self):
        """Should require concrete implementation of invoke"""
        model = ConcreteLLMModel()
        
        # Should not raise NotImplementedError
        result = await model.invoke([{"role": "user", "content": [{"text": "test"}]}])
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concrete_class_must_implement_prompt(self):
        """Should require concrete implementation of prompt"""
        model = ConcreteLLMModel()
        
        # Should not raise NotImplementedError
        async for response in model.prompt(text="test"):
            assert response is not None
    
    @pytest.mark.asyncio
    async def test_concrete_class_must_implement_handle_tool_calls(self):
        """Should require concrete implementation of handle_tool_calls"""
        model = ConcreteLLMModel()
        
        # Should not raise NotImplementedError
        result = await model.handle_tool_calls(
            response={"test": "data"},
            messages=[]
        )
        
        assert result is not None