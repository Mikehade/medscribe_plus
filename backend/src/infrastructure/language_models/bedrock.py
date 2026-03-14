"""
AWS Bedrock LLM provider implementation with async support.
Uses aioboto3 for async AWS SDK operations.
"""
import json
from typing import Any, AsyncGenerator, Dict, List, Optional
from botocore.config import Config
import aioboto3

from src.infrastructure.language_models.base import BaseLLMModel
from utils.logger import get_logger

logger = get_logger()

system_prompt: str = (
    "You are elle, a helpful and respectful personal assistant to "
    "{{ user_first_name | default(\"a user\") }} on Medscribe.\n"
    "Your responsibilities include but not limited to the following:\n\n"
    "- Help answer questions about MedScribe and how to carry out tasks.\n"
    "- Note taking and clinical documentation.\n"
    "- ICD Codes and more.\n\n"
    "- And more.\n"
)


class BedrockModel(BaseLLMModel):
    """
    AWS Bedrock model implementation with tool calling support.
    
    Supports models like:
    - nova lite
    - etc.
    """
    
    def __init__(
        self,
        aws_access_key: str,
        aws_secret_key: str,
        # model_id: str = "global.amazon.nova-2-lite-v1:0",
        # model_id: str = "amazon.nova-2-lite-v1:0",
        model_id: str = "arn:aws:bedrock:us-east-1:779056097161:inference-profile/us.amazon.nova-2-lite-v1:0",
        region_name: str = "us-east-1",
        temperature: float = 0.85,
        max_tokens: int = 4096,
        top_p: float = 0.6,
        stop_sequences: Optional[List[str]] = None,
        tool_registry=None,
        timezone_str: str = "UTC",
        **kwargs
    ):
        """
        Initialize Bedrock model.
        
        Args:
            aws_access_key: AWS access key ID
            aws_secret_key: AWS secret access key
            model_id: Bedrock model identifier
            region_name: AWS region
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop_sequences: List of stop sequences
            tool_registry: Tool registry for function calling
            timezone_str: Timezone for datetime rendering
        """
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            timezone_str=timezone_str,
            tool_registry=tool_registry,
            **kwargs
        )
        
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.region_name = region_name
        self.top_p = top_p
        self.stop_sequences = stop_sequences or ["python_tag"]
        self.reasoning = False
        self.current_tool_status = None  # Track current tool execution status
        self.tool_call_count = 0  # keep track of tool call count
        self.tool_tokens = 0      # keep track of tool call count
        
        # Configure boto3 with retries
        self.boto_config = Config(
            signature_version='v4',
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        
        # Create aioboto3 session
        self.session = aioboto3.Session(
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region_name
        )
    
    async def _get_client(self):
        """Get async Bedrock runtime client."""
        return self.session.client(
            'bedrock-runtime',
            config=self.boto_config
        )
    
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
        Invoke Bedrock model.
        
        Args:
            messages: Conversation messages in Bedrock format
            system: System prompt text
            stream: Whether to stream response
            enable_tools: Enable tool calling
            temperature: Override temperature
            max_tokens: Override max_tokens
            
        Returns:
            Model response dictionary
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Build payload
        payload = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "topP": self.top_p,
                "temperature": temp,
                "maxTokens": tokens,
                "stopSequences": self.stop_sequences
            }
        }

        if not self.reasoning:
            # logger.info(f"\n Messages going to model: {messages} \n")
            pass

        if self.reasoning:
            #only use top p when not reasoning
            payload["inferenceConfig"].pop("topP")
            # payload["guardrailConfig"] = {
            #     "guardrailIdentifier": 'arn:aws:bedrock:eu-central-1:961891917456:guardrail/32sxm7mbksd3',
            #     "guardrailVersion": "DRAFT",
            #     "trace": "disabled"
            # }
        
        # Add system prompt if provided
        if system:
            payload["system"] = [{"text": system}]
        
        # Add tool configuration if enabled
        if enable_tools and self.tool_registry:
            tool_config = self.tool_registry.generate_tool_config()
            if tool_config and tool_config.get("tools"):
                payload["toolConfig"] = tool_config
                logger.info(f"Tool config enabled with {len(tool_config.get('tools', []))} tools")
                # logger.info(f"Tools generated {payload} tools")
                logger.debug(f"Tools available: {[t['toolSpec']['name'] for t in tool_config.get('tools', [])]}")
            else:
                logger.warning("Tool registry exists but no tools were generated")
        elif enable_tools and not self.tool_registry:
            logger.warning("Tools enabled but no tool_registry configured")

        # enable response schema when needed
        if not enable_tools and kwargs.get("toolConfig"):
            payload["toolConfig"] = kwargs.get("toolConfig")
        
        # Log the payload (without sensitive data)
        logger.debug(f"Bedrock invoke payload keys: {payload.keys()}")
        # logger.debug(f"Bedrock invoke payload keys: {payload}")
        if "toolConfig" in payload:
            logger.debug(f"Number of tools in payload: {len(payload['toolConfig'].get('tools', []))}")
        
        # Invoke model
        async with await self._get_client() as client:
            if stream and not enable_tools:
                # Streaming not compatible with tool calling
                return await client.converse_stream(**payload)
            else:
                response = await client.converse(**payload)
                logger.debug(f"Bedrock response: {response.get('stopReason')}")
                return response
    
    async def prompt(
        self,
        text: Optional[str] = None,
        system_prompt: str =  system_prompt,
        system_context: Optional[Dict[str, Any]] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
        grammar: Dict | None = None,
        model_id: str = None,
        stream: bool = False,
        enable_tools: bool = True,
        tool_registry: list = [],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent_use: bool = True,
        reasoning: bool = False,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        High-level prompt interface with automatic tool handling.
        
        Args:
            text: User message
            system_prompt: System prompt template
            system_context: Variables for system prompt
            message_history: Previous messages
            grammar: json response schema for when we want response to follow a particluar pattern
            model_id: for when a particular model id is to be used
            stream: Stream response
            enable_tools: Enable tool calling
            temperature: Override temperature
            max_tokens: Override max_tokens
            
        Yields:
            Model responses
        """
        self.tool_call_count = 0
        self.tool_tokens = 0

        self.model_id = self.model_id if not model_id else model_id
        self.reasoning = reasoning # to keep track iof when not to use top_p
        # Render system prompt
        logger.info(f"Model ID in prompt: {self.model_id}")
        system = None
        if system_prompt:
            system = self.render_template(system_prompt, system_context or {})
        
        # Build messages
        messages = message_history.copy() if message_history else []
        
        if text:
            messages = self.clean_message_history(messages)
            messages.append({
                "role": "user",
                "content": [{"text": text}]
            })

        # if not self.reasoning:
        #     logger.info(f"\n Messages going to model: {messages} \n")
        
        if not messages:
            logger.error("No messages provided")
            raise ValueError("At least one message is required")

        # Log tool status
        if enable_tools:
            if self.tool_registry:
                available_tools = self.tool_registry.get_available_tools()
                logger.info(f"Tools enabled. Available tools: {available_tools}")
            else:
                logger.warning("Tools enabled but tool_registry is None")

        if not enable_tools and grammar:
            # logger.info(f"Response Schema: {grammar}")
            kwargs['toolConfig'] = grammar
        
        # Invoke model
        try:
            response = await self.invoke(
                messages=messages,
                system=system,
                stream=stream,
                enable_tools=enable_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # logger.info(f"\n Response in bedrock model prompt: {response} \n")

            # Track initial call tokens
            initial_tokens = response.get('usage', {}).get('totalTokens', 0)
            
            # Handle tool calls if present
            if enable_tools and self._has_tool_calls(response):
                logger.info("Tool calls detected, executing...")

                # will complete this later to track tool execution status
                # Yield status update
                # if self.current_tool_status:
                #     yield {
                #         "type": "tool_status",
                #         "status": self.current_tool_status
                #     }

                response = await self.handle_tool_calls(
                    response=response,
                    messages=messages,
                    system=system,
                    initial_tokens=initial_tokens,
                    **kwargs
                )
            # logger.info(f"\n About to yield response: {response} \n")

            # Inject tool metadata into response
            response['toolCallCount'] = self.tool_call_count
            response['toolTokens'] = self.tool_tokens
            yield response
            
        except Exception as e:
            logger.error(f"Error in Bedrock prompt: {e}", exc_info=True)
            raise
    
    async def handle_tool_calls(
        self,
        response: Dict[str, Any],
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_iterations: int = 15,
        initial_tokens: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle tool calls and continue conversation.
        
        Args:
            response: Initial model response with tool calls
            messages: Message history
            system: System prompt
            max_iterations: Maximum tool call iterations
            
        Returns:
            Final model response after tool execution
        """
        iteration = 0
        current_response = response
        
        while self._has_tool_calls(current_response) and iteration < max_iterations:
            iteration += 1
            logger.info(f"Tool call iteration {iteration}/{max_iterations}")

            # Extract any context text (but don't send to user - it breaks flow), will complete later
            context_text = self._extract_tool_context_text(current_response)
            if context_text:
                logger.debug(f"Model provided context: {context_text}")
            
            # Add assistant message with tool use
            assistant_message = {
                "role": "assistant",
                "content": current_response["output"]["message"]["content"]
            }
            messages.append(assistant_message)
            
            # Execute tool calls
            tool_results = []
            for content_block in current_response["output"]["message"]["content"]:
                if "toolUse" in content_block:
                    tool_use = content_block["toolUse"]
                    tool_name = tool_use["name"]

                    # Increment tool call count
                    self.tool_call_count += 1

                    # Generate user-friendly status message
                    # Store this in a class variable if you need to yield it
                    self.current_tool_status = f"Executing {tool_name}..." if not context_text else context_text
                    # logger.info(self.current_tool_status)

                    result = await self._execute_tool(tool_use)
                    tool_results.append(result)
            
            # Add tool results as user message
            user_message = {
                "role": "user",
                "content": tool_results
            }
            messages.append(user_message)
            
            # Continue conversation
            current_response = await self.invoke(
                messages=messages,
                system=system,
                enable_tools=True,
                **kwargs
            )

            # Accumulate tokens from this iteration
            usage = current_response.get('usage', {})
            iteration_tokens = usage.get('totalTokens', 0)
            self.tool_tokens += iteration_tokens
        
        if iteration >= max_iterations:
            logger.warning(f"Reached max tool call iterations ({max_iterations})")
        
        return current_response
    
    def _has_tool_calls(self, response: Dict[str, Any]) -> bool:
        """Check if response contains tool calls."""
        try:
            stop_reason = response.get("stopReason")
            if stop_reason == "tool_use":
                return True
            
            # Also check content blocks
            content = response.get("output", {}).get("message", {}).get("content", [])
            return any("toolUse" in block for block in content)
        except (KeyError, TypeError):
            return False

    def _extract_tool_context_text(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Extract text that appears alongside tool calls.
        This text is typically the model narrating what it's about to do.
        
        Returns:
            Text content if present, None otherwise
        """
        try:
            content = response.get("output", {}).get("message", {}).get("content", [])
            text_blocks = [block.get("text") for block in content if "text" in block]
            return " ".join(text_blocks).strip() if text_blocks else None
        except (KeyError, TypeError):
            return None
    
    async def _execute_tool(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool call.
        
        Args:
            tool_use: Tool use dictionary from model response
            
        Returns:
            Tool result dictionary
        """
        tool_use_id = tool_use["toolUseId"]
        tool_name = tool_use["name"]
        tool_input = tool_use["input"]
        
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool input: {tool_input}")
        
        try:
            # Execute tool through registry
            if self.tool_registry:
                output = await self.tool_registry.execute_tool(
                    tool_name=tool_name,
                    tool_input=tool_input
                )
            else:
                output = {"error": "No tool registry configured"}
            
            # Format result for Bedrock
            return {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": json.dumps(output)}]
                }
            }
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": json.dumps({"error": str(e)})}],
                    "status": "error"
                }
            }
    
    def extract_text_response(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from Bedrock response.
        
        Args:
            response: Bedrock API response
            
        Returns:
            Extracted text content
        """
        try:
            content_blocks = response["output"]["message"]["content"]
            
            # Concatenate all text blocks
            text_parts = []
            for block in content_blocks:
                if "text" in block:
                    text_parts.append(block["text"])
            
            return "\n".join(text_parts)
        except (KeyError, TypeError) as e:
            logger.error(f"Error extracting text: {e}")
            return ""