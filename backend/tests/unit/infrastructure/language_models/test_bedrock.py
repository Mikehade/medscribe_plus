"""
Unit tests for BedrockModel class.
Tests AWS Bedrock integration, tool calling, and async operations.
"""
import pytest
import json
from typing import Any, Dict, List
from botocore.exceptions import ClientError

from src.infrastructure.language_models.bedrock import BedrockModel


class TestBedrockModelInitialization:
    """Test suite for BedrockModel initialization"""
    
    def test_initializes_with_required_parameters(self):
        """Should initialize with AWS credentials and defaults"""
        model = BedrockModel(
            aws_access_key="test_key",
            aws_secret_key="test_secret"
        )
        
        assert model.aws_access_key == "test_key"
        assert model.aws_secret_key == "test_secret"
        assert model.region_name == "us-east-1"
        assert model.temperature == 0.85
        assert model.max_tokens == 4096
        assert model.top_p == 0.6
    
    def test_initializes_with_custom_parameters(
        self, 
        mocker
    ):
        """Should initialize with custom model and parameters"""
        mock_registry = mocker.Mock()
        
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.9,
            stop_sequences=["STOP"],
            tool_registry=mock_registry,
            timezone_str="America/New_York"
        )
        
        assert model.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert model.region_name == "us-east-1"
        assert model.temperature == 0.5
        assert model.max_tokens == 2048
        assert model.top_p == 0.9
        assert model.stop_sequences == ["STOP"]
        assert model.tool_registry == mock_registry
        assert model.timezone_str == "America/New_York"
    
    def test_initializes_boto_session(self):
        """Should create aioboto3 session with credentials"""
        model = BedrockModel(
            aws_access_key="test_key",
            aws_secret_key="test_secret"
        )
        
        assert model.session is not None
        assert model.boto_config is not None
    
    def test_sets_default_stop_sequences(self):
        """Should set default stop sequences"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        assert "python_tag" in model.stop_sequences
    
    def test_initializes_reasoning_flag_as_false(self):
        """Should initialize reasoning flag as False"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        assert model.reasoning is False


class TestInvoke:
    """Test suite for invoke method"""
    
    @pytest.mark.asyncio
    async def test_invokes_bedrock_converse_api(self, mocker):
        """Should call Bedrock converse API with correct payload"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        mock_client = mocker.AsyncMock()
        mock_client.converse = mocker.AsyncMock(return_value={
            "output": {
                "message": {
                    "content": [{"text": "Response"}]
                }
            },
            "stopReason": "end_turn"
        })
        
        mock_get_client = mocker.patch.object(
            model,
            '_get_client',
            return_value=mocker.AsyncMock(__aenter__=mocker.AsyncMock(return_value=mock_client))
        )
        
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        response = await model.invoke(messages=messages)
        
        assert response["stopReason"] == "end_turn"
        mock_client.converse.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_includes_system_prompt_in_payload(self, mocker):
        """Should include system prompt when provided"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        mock_client = mocker.AsyncMock()
        mock_client.converse = mocker.AsyncMock(return_value={
            "output": {"message": {"content": [{"text": "Response"}]}},
            "stopReason": "end_turn"
        })
        
        mocker.patch.object(
            model,
            '_get_client',
            return_value=mocker.AsyncMock(__aenter__=mocker.AsyncMock(return_value=mock_client))
        )
        
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        system = "You are a helpful assistant"
        
        await model.invoke(messages=messages, system=system)
        
        call_kwargs = mock_client.converse.call_args[1]
        assert "system" in call_kwargs
        assert call_kwargs["system"] == [{"text": system}]
    
    @pytest.mark.asyncio
    async def test_uses_override_temperature_and_max_tokens(self, mocker):
        """Should use override parameters when provided"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            temperature=0.7,
            max_tokens=1000
        )
        
        mock_client = mocker.AsyncMock()
        mock_client.converse = mocker.AsyncMock(return_value={
            "output": {"message": {"content": [{"text": "Response"}]}},
            "stopReason": "end_turn"
        })
        
        mocker.patch.object(
            model,
            '_get_client',
            return_value=mocker.AsyncMock(__aenter__=mocker.AsyncMock(return_value=mock_client))
        )
        
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        
        await model.invoke(
            messages=messages,
            temperature=0.5,
            max_tokens=2000
        )
        
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.5
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 2000
    
    @pytest.mark.asyncio
    async def test_includes_tool_config_when_enabled(self, mocker):
        """Should include tool configuration when tools enabled"""
        mock_registry = mocker.Mock()
        mock_registry.generate_tool_config.return_value = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "inputSchema": {"json": {}}
                    }
                }
            ]
        }
        
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            tool_registry=mock_registry
        )
        
        mock_client = mocker.AsyncMock()
        mock_client.converse = mocker.AsyncMock(return_value={
            "output": {"message": {"content": [{"text": "Response"}]}},
            "stopReason": "end_turn"
        })
        
        mocker.patch.object(
            model,
            '_get_client',
            return_value=mocker.AsyncMock(__aenter__=mocker.AsyncMock(return_value=mock_client))
        )
        
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        
        await model.invoke(messages=messages, enable_tools=True)
        
        call_kwargs = mock_client.converse.call_args[1]
        assert "toolConfig" in call_kwargs
        assert len(call_kwargs["toolConfig"]["tools"]) == 1
    
    @pytest.mark.asyncio
    async def test_does_not_include_tool_config_when_no_registry(self, mocker):
        """Should not include tools when registry is None"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            tool_registry=None
        )
        
        mock_client = mocker.AsyncMock()
        mock_client.converse = mocker.AsyncMock(return_value={
            "output": {"message": {"content": [{"text": "Response"}]}},
            "stopReason": "end_turn"
        })
        
        mocker.patch.object(
            model,
            '_get_client',
            return_value=mocker.AsyncMock(__aenter__=mocker.AsyncMock(return_value=mock_client))
        )
        
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        
        await model.invoke(messages=messages, enable_tools=True)
        
        call_kwargs = mock_client.converse.call_args[1]
        assert "toolConfig" not in call_kwargs
    
    @pytest.mark.asyncio
    async def test_omits_top_p_when_reasoning_enabled(self, mocker):
        """Should not include top_p when reasoning is True"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        model.reasoning = True
        
        mock_client = mocker.AsyncMock()
        mock_client.converse = mocker.AsyncMock(return_value={
            "output": {"message": {"content": [{"text": "Response"}]}},
            "stopReason": "end_turn"
        })
        
        mocker.patch.object(
            model,
            '_get_client',
            return_value= mocker.AsyncMock(__aenter__=mocker.AsyncMock(return_value=mock_client))
        )
        
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        
        await model.invoke(messages=messages)
        
        call_kwargs = mock_client.converse.call_args[1]
        assert "topP" not in call_kwargs["inferenceConfig"]


class TestPrompt:
    """Test suite for prompt method"""
    
    @pytest.mark.asyncio
    async def test_yields_response_from_invoke(self, mocker):
        """Should yield response from invoke method"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        mock_response = {
            "output": {
                "message": {
                    "content": [{"text": "Test response"}]
                }
            },
            "stopReason": "end_turn"
        }
        
        mocker.patch.object(model, 'invoke', return_value=mock_response)
        
        responses = []
        async for response in model.prompt(text="Hello"):
            responses.append(response)
        
        assert len(responses) == 1
        assert responses[0] == mock_response
    
    @pytest.mark.asyncio
    async def test_renders_system_prompt_with_context(self, mocker):
        """Should render system prompt template with context"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        mock_response = {
            "output": {"message": {"content": [{"text": "Response"}]}},
            "stopReason": "end_turn"
        }
        
        mock_invoke = mocker.patch.object(model, 'invoke', return_value=mock_response)
        
        system_prompt = "Hello {{ user_first_name }}"
        system_context = {"user_first_name": "Alice"}
        
        async for _ in model.prompt(
            text="Test",
            system_prompt=system_prompt,
            system_context=system_context
        ):
            pass
        
        # Check that invoke was called with rendered system prompt
        call_kwargs = mock_invoke.call_args[1]
        assert "Hello Alice" in call_kwargs["system"]
    
    @pytest.mark.asyncio
    async def test_cleans_message_history(self, mocker):
        """Should clean message history before sending"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        mock_response = {
            "output": {"message": {"content": [{"text": "Response"}]}},
            "stopReason": "end_turn"
        }
        
        mock_invoke = mocker.patch.object(model, 'invoke', return_value=mock_response)
        mock_clean = mocker.patch.object(model, 'clean_message_history', return_value=[])
        
        message_history = [
            {"role": "user", "content": [{"text": "Hi"}]},
            {"role": "assistant", "content": [{"text": "ok"}]}  # Trivial
        ]
        
        async for _ in model.prompt(
            text="New message",
            message_history=message_history
        ):
            pass
        
        mock_clean.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_appends_new_user_message(self, mocker):
        """Should append new user message to history"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        mock_response = {
            "output": {"message": {"content": [{"text": "Response"}]}},
            "stopReason": "end_turn"
        }
        
        mock_invoke = mocker.patch.object(model, 'invoke', return_value=mock_response)
        
        async for _ in model.prompt(text="New message"):
            pass
        
        messages = mock_invoke.call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == "New message"
    
    @pytest.mark.asyncio
    async def test_handles_tool_calls_when_detected(self, mocker):
        """Should handle tool calls when they are present"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        tool_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "123",
                                "name": "test_tool",
                                "input": {}
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use"
        }
        
        final_response = {
            "output": {"message": {"content": [{"text": "Final response"}]}},
            "stopReason": "end_turn"
        }
        
        mocker.patch.object(model, 'invoke', return_value=tool_response)
        mock_handle_tools = mocker.patch.object(
            model,
            'handle_tool_calls',
            return_value=final_response
        )
        
        responses = []
        async for response in model.prompt(text="Test", enable_tools=True):
            responses.append(response)
        
        mock_handle_tools.assert_called_once()
        assert responses[0] == final_response
    
    @pytest.mark.asyncio
    async def test_raises_error_when_no_messages(self, mocker):
        """Should raise ValueError when no messages provided"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        with pytest.raises(ValueError, match="At least one message is required"):
            async for _ in model.prompt(text=None, message_history=None):
                pass
    
    @pytest.mark.asyncio
    async def test_sets_reasoning_flag_when_enabled(self, mocker):
        """Should set reasoning flag when reasoning=True"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        assert model.reasoning is False
        
        mock_response = {
            "output": {"message": {"content": [{"text": "Response"}]}},
            "stopReason": "end_turn"
        }
        
        mocker.patch.object(model, 'invoke', return_value=mock_response)
        
        async for _ in model.prompt(text="Test", reasoning=True):
            pass
        
        assert model.reasoning is True


class TestHandleToolCalls:
    """Test suite for handle_tool_calls method"""
    
    @pytest.mark.asyncio
    async def test_executes_single_tool_call(self, mocker):
        """Should execute single tool and continue conversation"""
        mock_registry = mocker.Mock()
        mock_registry.execute_tool = mocker.AsyncMock(return_value={"result": "success"})
        
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            tool_registry=mock_registry
        )
        
        initial_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tool-123",
                                "name": "test_tool",
                                "input": {"param": "value"}
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use"
        }
        
        final_response = {
            "output": {"message": {"content": [{"text": "Final answer"}]}},
            "stopReason": "end_turn"
        }
        
        mock_invoke = mocker.patch.object(
            model,
            'invoke',
            return_value=final_response
        )
        
        result = await model.handle_tool_calls(
            response=initial_response,
            messages=[],
            system="System prompt"
        )
        
        assert result == final_response
        mock_registry.execute_tool.assert_called_once_with(
            tool_name="test_tool",
            tool_input={"param": "value"}
        )
    
    @pytest.mark.asyncio
    async def test_handles_multiple_tool_calls(self, mocker):
        """Should handle multiple tool calls in one response"""
        mock_registry = mocker.Mock()
        mock_registry.execute_tool = mocker.AsyncMock(return_value={"result": "success"})
        
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            tool_registry=mock_registry
        )
        
        initial_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tool-1",
                                "name": "tool_one",
                                "input": {}
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "tool-2",
                                "name": "tool_two",
                                "input": {}
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use"
        }
        
        final_response = {
            "output": {"message": {"content": [{"text": "Final"}]}},
            "stopReason": "end_turn"
        }
        
        mocker.patch.object(model, 'invoke', return_value=final_response)
        
        await model.handle_tool_calls(
            response=initial_response,
            messages=[]
        )
        
        assert mock_registry.execute_tool.call_count == 2
    
    @pytest.mark.asyncio
    async def test_respects_max_iterations(self, mocker):
        """Should stop after max_iterations"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        tool_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "123",
                                "name": "test_tool",
                                "input": {}
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use"
        }
        
        # Always return tool_use to test max iterations
        mock_invoke = mocker.patch.object(
            model,
            'invoke',
            return_value=tool_response
        )
        
        mock_execute = mocker.patch.object(
            model,
            '_execute_tool',
            return_value={"toolResult": {"toolUseId": "123", "content": [{"text": "{}"}]}}
        )
        
        result = await model.handle_tool_calls(
            response=tool_response,
            messages=[],
            max_iterations=3
        )
        
        # Should call invoke 3 times (for continuation after each tool call)
        assert mock_invoke.call_count == 3
    
    @pytest.mark.asyncio
    async def test_adds_assistant_and_user_messages(self, mocker):
        """Should add assistant message with tool use and user message with results"""
        mock_registry = mocker.Mock()
        mock_registry.execute_tool = mocker.AsyncMock(return_value={"result": "data"})
        
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            tool_registry=mock_registry
        )
        
        initial_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "123",
                                "name": "test_tool",
                                "input": {}
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use"
        }
        
        final_response = {
            "output": {"message": {"content": [{"text": "Done"}]}},
            "stopReason": "end_turn"
        }
        
        mock_invoke = mocker.patch.object(model, 'invoke', return_value=final_response)
        
        messages = []
        await model.handle_tool_calls(
            response=initial_response,
            messages=messages
        )
        
        # Should have added assistant and user messages
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"


class TestExecuteTool:
    """Test suite for _execute_tool method"""
    
    @pytest.mark.asyncio
    async def test_executes_tool_successfully(self, mocker):
        """Should execute tool and return formatted result"""
        mock_registry = mocker.Mock()
        mock_registry.execute_tool = mocker.AsyncMock(return_value={"status": "success", "data": "result"})
        
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            tool_registry=mock_registry
        )
        
        tool_use = {
            "toolUseId": "tool-123",
            "name": "test_tool",
            "input": {"param": "value"}
        }
        
        result = await model._execute_tool(tool_use)
        
        assert "toolResult" in result
        assert result["toolResult"]["toolUseId"] == "tool-123"
        assert "content" in result["toolResult"]
        
        # Content should be JSON string
        content_text = result["toolResult"]["content"][0]["text"]
        parsed = json.loads(content_text)
        assert parsed["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_handles_tool_execution_error(self, mocker):
        """Should handle and return error when tool fails"""
        mock_registry = mocker.Mock()
        mock_registry.execute_tool = mocker.AsyncMock(side_effect=Exception("Tool failed"))
        
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            tool_registry=mock_registry
        )
        
        tool_use = {
            "toolUseId": "tool-123",
            "name": "failing_tool",
            "input": {}
        }
        
        result = await model._execute_tool(tool_use)
        
        assert "toolResult" in result
        assert result["toolResult"]["status"] == "error"
        content_text = result["toolResult"]["content"][0]["text"]
        parsed = json.loads(content_text)
        assert "error" in parsed
    
    @pytest.mark.asyncio
    async def test_returns_error_when_no_registry(self):
        """Should return error when no tool registry configured"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret",
            tool_registry=None
        )
        
        tool_use = {
            "toolUseId": "tool-123",
            "name": "test_tool",
            "input": {}
        }
        
        result = await model._execute_tool(tool_use)
        
        content_text = result["toolResult"]["content"][0]["text"]
        parsed = json.loads(content_text)
        assert "error" in parsed
        assert "No tool registry configured" in parsed["error"]


class TestHasToolCalls:
    """Test suite for _has_tool_calls method"""
    
    def test_detects_tool_use_stop_reason(self):
        """Should detect tool_use stop reason"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        response = {"stopReason": "tool_use"}
        
        assert model._has_tool_calls(response) is True
    
    def test_detects_tool_use_in_content(self):
        """Should detect toolUse in content blocks"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        response = {
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"name": "test"}}
                    ]
                }
            }
        }
        
        assert model._has_tool_calls(response) is True
    
    def test_returns_false_for_text_only_response(self):
        """Should return False for text-only response"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        response = {
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "content": [{"text": "Response"}]
                }
            }
        }
        
        assert model._has_tool_calls(response) is False
    
    def test_handles_malformed_response(self):
        """Should handle malformed response gracefully"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        response = {"invalid": "structure"}
        
        assert model._has_tool_calls(response) is False


class TestExtractTextResponse:
    """Test suite for extract_text_response method"""
    
    def test_extracts_single_text_block(self):
        """Should extract text from single content block"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        response = {
            "output": {
                "message": {
                    "content": [{"text": "Hello world"}]
                }
            }
        }
        
        result = model.extract_text_response(response)
        
        assert result == "Hello world"
    
    def test_concatenates_multiple_text_blocks(self):
        """Should concatenate multiple text blocks"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "First part"},
                        {"text": "Second part"}
                    ]
                }
            }
        }
        
        result = model.extract_text_response(response)
        
        assert result == "First part\nSecond part"
    
    def test_ignores_non_text_blocks(self):
        """Should ignore non-text content blocks"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Text content"},
                        {"toolUse": {"name": "tool"}}
                    ]
                }
            }
        }
        
        result = model.extract_text_response(response)
        
        assert result == "Text content"
    
    def test_returns_empty_string_for_malformed_response(self):
        """Should return empty string for malformed response"""
        model = BedrockModel(
            aws_access_key="key",
            aws_secret_key="secret"
        )
        
        response = {"invalid": "structure"}
        
        result = model.extract_text_response(response)
        
        assert result == ""