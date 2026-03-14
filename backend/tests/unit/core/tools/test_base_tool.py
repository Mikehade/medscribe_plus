"""
Unit tests for BaseTool and ToolRegistry classes.
Tests focus on behavior rather than implementation details.
"""
import pytest
from typing import Any, Dict
from src.core.tools.base import BaseTool, ToolRegistry


# Concrete implementation of BaseTool for testing
class ConcreteTool(BaseTool):
    """Concrete tool implementation for testing"""
    
    def __init__(
        self, 
        enabled_tools=None, 
        **kwargs
    ):
        super().__init__(enabled_tools=enabled_tools, **kwargs)
        self.execution_count = 0
    
    async def execute(
        self, tool_name: str, 
        tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool method"""
        method = self.get_tool_method(tool_name)
        if not method:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        self.execution_count += 1
        return await method(**tool_input)
    
    async def _test_tool(
        self, 
        param1: str, 
        param2: int = 10
    ) -> Dict[str, Any]:
        """Test tool for demonstration"""
        return {"success": True, "data": {"param1": param1, "param2": param2}}
    
    async def _another_tool(
        self, 
        required_param: str
    ) -> Dict[str, Any]:
        """Another test tool"""
        return {"success": True, "data": {"required_param": required_param}}
    
    async def _disabled_tool(
        self
    ) -> Dict[str, Any]:
        """Tool that might be disabled"""
        return {"success": True, "data": "disabled_result"}


class TestBaseToolInitialization:
    """Test suite for BaseTool initialization"""
    
    def test_tool_initializes_with_no_kwargs(
        self
    ):
        """Should initialize with no additional kwargs"""
        tool = ConcreteTool()
        assert tool.kwargs == {}
        assert tool.enabled_tools is None
    
    def test_tool_initializes_with_kwargs(
        self
    ):
        """Should store kwargs for later use"""
        tool = ConcreteTool(test_param="value", another_param=123)
        assert tool.kwargs["test_param"] == "value"
        assert tool.kwargs["another_param"] == 123
    
    def test_tool_initializes_with_enabled_tools_list(
        self
    ):
        """Should initialize with specific enabled tools"""
        tool = ConcreteTool(enabled_tools=["test_tool", "another_tool"])
        assert tool.enabled_tools == ["test_tool", "another_tool"]


class TestGenerateBedrockConfig:
    """Test suite for generate_bedrock_config method"""
    
    def test_generates_config_for_all_tools_when_none_specified(
        self
    ):
        """Should generate config for all available tools"""
        config = ConcreteTool.generate_bedrock_config()
        
        assert "tools" in config
        assert "toolChoice" in config
        assert isinstance(config["tools"], list)
        assert len(config["tools"]) == 3  # test_tool, another_tool, disabled_tool
    
    def test_generates_config_for_specific_tools_only(
        self
    ):
        """Should generate config only for enabled tools"""
        config = ConcreteTool.generate_bedrock_config(
            enabled_tools=["test_tool"]
        )
        
        tool_names = [tool["toolSpec"]["name"] for tool in config["tools"]]
        assert "test_tool" in tool_names
        assert "another_tool" not in tool_names
        assert "disabled_tool" not in tool_names
    
    def test_tool_spec_has_correct_structure(
        self
    ):
        """Should generate tool spec with correct Bedrock structure"""
        config = ConcreteTool.generate_bedrock_config(
            enabled_tools=["test_tool"]
        )
        
        tool_spec = config["tools"][0]["toolSpec"]
        assert "name" in tool_spec
        assert "description" in tool_spec
        assert "inputSchema" in tool_spec
        assert tool_spec["name"] == "test_tool"
    
    def test_extracts_parameters_correctly(
        self
    ):
        """Should extract method parameters into tool spec"""
        config = ConcreteTool.generate_bedrock_config(
            enabled_tools=["test_tool"]
        )
        
        properties = config["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]
        required = config["tools"][0]["toolSpec"]["inputSchema"]["json"]["required"]
        
        assert "param1" in properties
        assert "param2" in properties
        assert "param1" in required  # No default value
        assert "param2" not in required  # Has default value
    
    def test_excludes_dunder_methods(
        self
    ):
        """Should not include __init__, __str__, etc."""
        config = ConcreteTool.generate_bedrock_config()
        
        tool_names = [tool["toolSpec"]["name"] for tool in config["tools"]]
        assert "__init__" not in tool_names
        assert "__str__" not in tool_names


class TestGenerateOpenAIConfig:
    """Test suite for generate_openai_config method"""
    
    def test_generates_openai_functions_for_all_tools(
        self
    ):
        """Should generate OpenAI function config for all tools"""
        functions = ConcreteTool.generate_openai_config()
        
        assert isinstance(functions, list)
        assert len(functions) == 3
    
    def test_generates_functions_for_specific_tools_only(
        self
    ):
        """Should generate functions only for enabled tools"""
        functions = ConcreteTool.generate_openai_config(
            enabled_tools=["another_tool"]
        )
        
        function_names = [func["name"] for func in functions]
        assert "another_tool" in function_names
        assert "test_tool" not in function_names
    
    def test_function_has_correct_openai_structure(
        self
    ):
        """Should generate function with OpenAI-compatible structure"""
        functions = ConcreteTool.generate_openai_config(
            enabled_tools=["test_tool"]
        )
        
        func = functions[0]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        assert func["parameters"]["type"] == "object"
        assert "properties" in func["parameters"]
        assert "required" in func["parameters"]


class TestGetToolMethod:
    """Test suite for get_tool_method"""
    
    def test_returns_method_for_valid_tool_name(
        self
    ):
        """Should return method when tool exists"""
        tool = ConcreteTool()
        method = tool.get_tool_method("test_tool")
        
        assert method is not None
        assert callable(method)
    
    def test_returns_none_for_invalid_tool_name(
        self
    ):
        """Should return None when tool doesn't exist"""
        tool = ConcreteTool()
        method = tool.get_tool_method("nonexistent_tool")
        
        assert method is None
    
    def test_method_can_be_called(
        self
    ):
        """Should return callable method that can be executed"""
        tool = ConcreteTool()
        method = tool.get_tool_method("test_tool")
        
        # Method should be awaitable
        import inspect
        assert inspect.iscoroutinefunction(method)


class TestBaseToolExecution:
    """Test suite for execute method"""
    
    @pytest.mark.asyncio
    async def test_executes_tool_successfully(
        self
    ):
        """Should execute tool and return result"""
        tool = ConcreteTool()
        
        result = await tool.execute(
            "test_tool",
            {"param1": "value", "param2": 20}
        )
        
        assert result["success"] is True
        assert result["data"]["param1"] == "value"
        assert result["data"]["param2"] == 20
    
    @pytest.mark.asyncio
    async def test_returns_error_for_nonexistent_tool(
        self
    ):
        """Should return error when tool doesn't exist"""
        tool = ConcreteTool()
        
        result = await tool.execute("invalid_tool", {})
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_increments_execution_count(
        self
    ):
        """Should track execution count"""
        tool = ConcreteTool()
        assert tool.execution_count == 0
        
        await tool.execute("test_tool", {"param1": "test"})
        assert tool.execution_count == 1
        
        await tool.execute("another_tool", {"required_param": "test"})
        assert tool.execution_count == 2


class TestToolRegistryInitialization:
    """Test suite for ToolRegistry initialization"""
    
    def test_initializes_empty_registry(
        self
    ):
        """Should initialize with empty tool list"""
        registry = ToolRegistry()
        
        assert registry.tool_classes == []
        assert len(registry._tool_map) == 0
    
    def test_initializes_with_tool_classes(
        self
    ):
        """Should initialize and build tool map from provided tools"""
        tool1 = ConcreteTool()
        tool2 = ConcreteTool()
        
        registry = ToolRegistry(tool_classes=[tool1, tool2])
        
        assert len(registry.tool_classes) == 2
        assert len(registry._tool_map) > 0
    
    def test_builds_tool_map_correctly(
        self
    ):
        """Should map tool names to tool instances"""
        tool = ConcreteTool(enabled_tools=["test_tool"])
        registry = ToolRegistry(tool_classes=[tool])
        
        assert "test_tool" in registry._tool_map
        assert registry._tool_map["test_tool"] == tool
    
    def test_respects_enabled_tools_filter(
        self
    ):
        """Should only include enabled tools in map"""
        tool = ConcreteTool(enabled_tools=["test_tool"])
        registry = ToolRegistry(tool_classes=[tool])
        
        assert "test_tool" in registry._tool_map
        assert "another_tool" not in registry._tool_map
        assert "disabled_tool" not in registry._tool_map


class TestToolRegistryGenerateConfig:
    """Test suite for generate_tool_config method"""
    
    def test_returns_empty_config_for_empty_registry(
        self
    ):
        """Should return empty dict when no tools registered"""
        registry = ToolRegistry()
        config = registry.generate_tool_config()
        
        assert config == {}
    
    def test_generates_combined_config_from_multiple_tools(
        self
    ):
        """Should combine configs from all registered tools"""
        tool1 = ConcreteTool(enabled_tools=["test_tool"])
        tool2 = ConcreteTool(enabled_tools=["another_tool"])
        
        registry = ToolRegistry(tool_classes=[tool1, tool2])
        config = registry.generate_tool_config()
        
        assert "tools" in config
        assert len(config["tools"]) == 2
    
    def test_config_has_bedrock_structure(
        self
    ):
        """Should generate config with Bedrock structure"""
        tool = ConcreteTool()
        registry = ToolRegistry(tool_classes=[tool])
        
        config = registry.generate_tool_config()
        
        assert "tools" in config
        assert "toolChoice" in config
        assert "auto" in config["toolChoice"]


class TestToolRegistryGenerateOpenAIFunctions:
    """Test suite for generate_openai_functions method"""
    
    def test_returns_empty_list_for_empty_registry(
        self
    ):
        """Should return empty list when no tools registered"""
        registry = ToolRegistry()
        functions = registry.generate_openai_functions()
        
        assert functions == []
    
    def test_generates_combined_functions_from_multiple_tools(
        self
    ):
        """Should combine functions from all registered tools"""
        tool1 = ConcreteTool(enabled_tools=["test_tool"])
        tool2 = ConcreteTool(enabled_tools=["another_tool"])
        
        registry = ToolRegistry(tool_classes=[tool1, tool2])
        functions = registry.generate_openai_functions()
        
        assert len(functions) == 2
        function_names = [f["name"] for f in functions]
        assert "test_tool" in function_names
        assert "another_tool" in function_names


class TestToolRegistryExecuteTool:
    """Test suite for execute_tool method"""
    
    @pytest.mark.asyncio
    async def test_executes_registered_tool_successfully(
        self
    ):
        """Should execute tool from registry"""
        tool = ConcreteTool()
        registry = ToolRegistry(tool_classes=[tool])
        
        result = await registry.execute_tool(
            "test_tool",
            {"param1": "value", "param2": 15}
        )
        
        assert result["success"] is True
        assert result["data"]["param1"] == "value"
    
    @pytest.mark.asyncio
    async def test_returns_error_for_unregistered_tool(
        self
    ):
        """Should return error when tool not in registry"""
        registry = ToolRegistry()
        
        result = await registry.execute_tool("nonexistent_tool", {})
        
        assert result["success"] is False
        assert "not found in registry" in result["error"]
    
    @pytest.mark.asyncio
    async def test_handles_tool_execution_exception(
        self
    ):
        """Should catch and return error when tool execution fails"""
        # Create a tool that will raise an exception
        class FailingTool(BaseTool):
            async def execute(self, tool_name: str, tool_input: Dict[str, Any]):
                raise ValueError("Intentional failure")
            
            async def _failing_tool(self):
                """Tool that fails"""
                pass
        
        tool = FailingTool()
        registry = ToolRegistry(tool_classes=[tool])
        
        result = await registry.execute_tool("failing_tool", {})
        
        assert result["success"] is False
        assert "error" in result


class TestToolRegistryAddToolClass:
    """Test suite for add_tool_class method"""
    
    def test_adds_new_tool_to_registry(
        self
    ):
        """Should add tool class to registry"""
        registry = ToolRegistry()
        assert len(registry.tool_classes) == 0
        
        tool = ConcreteTool()
        registry.add_tool_class(tool)
        
        assert len(registry.tool_classes) == 1
        assert tool in registry.tool_classes
    
    def test_rebuilds_tool_map_after_adding(
        self
    ):
        """Should rebuild tool map after adding new tool"""
        registry = ToolRegistry()
        assert len(registry._tool_map) == 0
        
        tool = ConcreteTool(enabled_tools=["test_tool"])
        registry.add_tool_class(tool)
        
        assert "test_tool" in registry._tool_map


class TestToolRegistryGetAvailableTools:
    """Test suite for get_available_tools method"""
    
    def test_returns_empty_list_for_empty_registry(
        self
    ):
        """Should return empty list when no tools registered"""
        registry = ToolRegistry()
        tools = registry.get_available_tools()
        
        assert tools == []
    
    def test_returns_all_available_tool_names(
        self
    ):
        """Should return list of all tool names"""
        tool = ConcreteTool()
        registry = ToolRegistry(tool_classes=[tool])
        
        tools = registry.get_available_tools()
        
        assert "test_tool" in tools
        assert "another_tool" in tools
        assert "disabled_tool" in tools
    
    def test_returns_only_enabled_tools(
        self
    ):
        """Should return only enabled tools when filter applied"""
        tool = ConcreteTool(enabled_tools=["test_tool", "another_tool"])
        registry = ToolRegistry(tool_classes=[tool])
        
        tools = registry.get_available_tools()
        
        assert "test_tool" in tools
        assert "another_tool" in tools
        assert "disabled_tool" not in tools
    
    def test_returns_tools_from_multiple_tool_classes(
        self
    ):
        """Should return tools from all registered classes"""
        tool1 = ConcreteTool(enabled_tools=["test_tool"])
        tool2 = ConcreteTool(enabled_tools=["another_tool"])
        
        registry = ToolRegistry(tool_classes=[tool1, tool2])
        tools = registry.get_available_tools()
        
        assert len(tools) == 2
        assert "test_tool" in tools
        assert "another_tool" in tools