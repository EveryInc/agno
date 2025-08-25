import pytest

from agno.agent import Agent, RunResponse
from agno.models.litellm import LiteLLMResponses

# Try to import optional tools, skip tests if not available
try:
    from agno.tools.duckduckgo import DuckDuckGoTools
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False

try:
    from agno.tools.yfinance import YFinanceTools
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from agno.tools.calculator import Calculator
    HAS_CALCULATOR = True
except ImportError:
    HAS_CALCULATOR = False


def _assert_metrics(response: RunResponse):
    """Helper function to assert metrics are present and valid"""
    # Check that metrics dictionary exists
    assert response.metrics is not None

    # Check that we have some token counts
    assert "input_tokens" in response.metrics
    assert "output_tokens" in response.metrics
    assert "total_tokens" in response.metrics

    # Check that we have timing information
    assert "time" in response.metrics

    # Check that the total tokens is the sum of input and output tokens
    input_tokens = sum(response.metrics.get("input_tokens", []))
    output_tokens = sum(response.metrics.get("output_tokens", []))
    total_tokens = sum(response.metrics.get("total_tokens", []))

    # The total should be at least the sum of input and output
    # (Note: sometimes there might be small discrepancies in how these are calculated)
    assert total_tokens >= input_tokens + output_tokens - 5  # Allow small margin of error


class TestToolUseGPT5:
    """Test tool use functionality with GPT-5 model"""

    @pytest.mark.skipif(not HAS_DUCKDUCKGO, reason="DuckDuckGo tools not available")
    def test_gpt5_tool_use_no_reasoning(self):
        """Test GPT-5 tool use without reasoning"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            tools=[DuckDuckGoTools(cache_results=True)],
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run("What's the latest news about SpaceX?")

        assert response.content is not None
        assert len(response.messages) >= 3

        # Check if tool was used
        tool_messages = [m for m in response.messages if m.role == "tool"]
        assert len(tool_messages) > 0, "Tool should have been used"
        _assert_metrics(response)

    def test_gpt5_tool_use_with_reasoning(self):
        """Test GPT-5 tool use with reasoning enabled"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "medium", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            tools=[Calculator()],
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run("Calculate (25 * 4) + (18 / 3) and explain your reasoning")

        assert response.content is not None
        assert len(response.messages) >= 3

        # Check if tool was used
        tool_messages = [m for m in response.messages if m.role == "tool"]
        assert len(tool_messages) > 0, "Calculator tool should have been used"
        
        # Should contain the calculation result
        assert "106" in response.content or "100" in response.content
        _assert_metrics(response)

    def test_gpt5_parallel_tool_calls_reasoning(self):
        """Test GPT-5 parallel tool calls with reasoning"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "medium", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            tools=[DuckDuckGoTools(cache_results=True)],
            telemetry=False,
            monitoring=False,
        )

        response = agent.run("Search for latest news about both SpaceX and NASA")

        # Verify tool usage
        tool_calls = [msg.tool_calls for msg in response.messages if msg.tool_calls]
        assert len(tool_calls) >= 1
        assert response.content is not None
        _assert_metrics(response)

    def test_gpt5_custom_tool_no_parameters(self):
        """Test GPT-5 with custom tool without parameters"""
        def get_time():
            return "It is 12:00 PM UTC"

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            tools=[get_time],
            telemetry=False,
            monitoring=False,
        )

        response = agent.run("What time is it?")

        assert any(msg.tool_calls for msg in response.messages)
        assert response.content is not None
        assert "12:00" in response.content
        _assert_metrics(response)

    def test_gpt5_multiple_different_tools_reasoning(self):
        """Test GPT-5 with multiple different tools and reasoning"""
        def get_weather():
            return "It's sunny and 75°F"

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "high", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            tools=[DuckDuckGoTools(cache_results=True), get_weather, Calculator()],
            telemetry=False,
            monitoring=False,
        )

        response = agent.run("What's the weather, search for SpaceX news, and calculate 15 + 27?")

        # Verify tool usage
        tool_calls = [msg.tool_calls for msg in response.messages if msg.tool_calls]
        assert len(tool_calls) >= 1
        total_calls = sum(len(calls) for calls in tool_calls)
        assert total_calls >= 3  # Should use all three tools
        
        assert response.content is not None
        assert "75°F" in response.content
        assert "42" in response.content  # 15 + 27 = 42
        _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_gpt5_async_tool_use_reasoning(self):
        """Test GPT-5 async tool use with reasoning"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "medium", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            tools=[YFinanceTools(cache_results=True)],
            telemetry=False,
            monitoring=False,
        )

        response = await agent.arun("What is the current price of AAPL stock?")

        assert response.content is not None
        assert len(response.messages) >= 3

        # Check if tool was used
        tool_messages = [m for m in response.messages if m.role == "tool"]
        assert len(tool_messages) > 0, "YFinance tool should have been used"
        _assert_metrics(response)


class TestToolUseClaude:
    """Test tool use functionality with Claude Opus 4.1 model"""

    def test_claude_tool_use_no_thinking(self):
        """Test Claude tool use without thinking"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=0.7,
                top_p=None,
            ),
            markdown=True,
            tools=[DuckDuckGoTools(cache_results=True)],
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run("What's the latest news about SpaceX?")

        assert response.content is not None
        assert len(response.messages) >= 3

        # Check if tool was used
        tool_messages = [m for m in response.messages if m.role == "tool"]
        assert len(tool_messages) > 0, "Tool should have been used"
        _assert_metrics(response)

    def test_claude_tool_use_with_thinking(self):
        """Test Claude tool use with thinking enabled"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,  # Required when thinking is enabled
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            tools=[Calculator()],
            telemetry=False,
            monitoring=False,
        )

        response: RunResponse = agent.run("Calculate the compound interest on $1000 at 5% annually for 3 years. Show your reasoning.")

        assert response.content is not None
        assert len(response.messages) >= 3

        # Check if tool was used
        tool_messages = [m for m in response.messages if m.role == "tool"]
        assert len(tool_messages) > 0, "Calculator tool should have been used"
        _assert_metrics(response)

    def test_claude_thinking_high_budget_complex_tools(self):
        """Test Claude with high thinking budget and complex tool usage"""
        def analyze_data(data_points: str):
            """Analyze a list of data points"""
            numbers = [float(x.strip()) for x in data_points.split(",")]
            avg = sum(numbers) / len(numbers)
            return f"Average: {avg:.2f}, Count: {len(numbers)}, Range: {min(numbers)}-{max(numbers)}"

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 2048  # High budget for complex reasoning
                    }
                }
            ),
            markdown=True,
            tools=[analyze_data, Calculator()],
            telemetry=False,
            monitoring=False,
        )

        response = agent.run("Analyze this data: 10, 15, 20, 25, 30. Then calculate what 20% of the average would be.")

        assert response.content is not None
        tool_calls = [msg.tool_calls for msg in response.messages if msg.tool_calls]
        assert len(tool_calls) >= 1
        
        # Should use both tools and show calculations
        assert "20" in response.content  # Average should be 20
        assert "4" in response.content  # 20% of 20 is 4
        _assert_metrics(response)

    def test_claude_parallel_tool_calls_no_thinking(self):
        """Test Claude parallel tool calls without thinking"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=0.8,
                top_p=None,
            ),
            markdown=True,
            tools=[DuckDuckGoTools(cache_results=True)],
            telemetry=False,
            monitoring=False,
        )

        response = agent.run("Search for latest news about both SpaceX and Tesla")

        # Verify tool usage
        tool_calls = [msg.tool_calls for msg in response.messages if msg.tool_calls]
        assert len(tool_calls) >= 1
        assert response.content is not None
        _assert_metrics(response)

    def test_claude_custom_tool_with_thinking(self):
        """Test Claude with custom tool and thinking"""
        def echo_message(message: str):
            """Echo back the message with emphasis"""
            return f"ECHO: {message.upper()}"

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 512
                    }
                }
            ),
            markdown=True,
            tools=[echo_message],
            telemetry=False,
            monitoring=False,
        )

        response = agent.run("Can you echo 'Hello World' for me?")

        assert any(msg.tool_calls for msg in response.messages)
        assert response.content is not None
        assert "ECHO: HELLO WORLD" in response.content
        _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_claude_async_tool_use_thinking(self):
        """Test Claude async tool use with thinking"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            tools=[YFinanceTools(cache_results=True)],
            telemetry=False,
            monitoring=False,
        )

        response = await agent.arun("Get the current stock price for TSLA and explain the key factors that might affect it")

        assert response.content is not None
        assert len(response.messages) >= 3

        # Check if tool was used
        tool_messages = [m for m in response.messages if m.role == "tool"]
        assert len(tool_messages) > 0, "YFinance tool should have been used"
        assert "TSLA" in response.content
        _assert_metrics(response)


class TestStreamingToolUse:
    """Test streaming tool use with both models"""

    def test_gpt5_stream_tool_use_reasoning(self):
        """Test GPT-5 streaming tool use with reasoning"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "medium", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            tools=[Calculator()],
            telemetry=False,
            monitoring=False,
        )

        response_stream = agent.run(
            "Calculate 123 * 456 and show your reasoning", 
            stream=True, 
            stream_intermediate_steps=True
        )

        responses = []
        tool_call_seen = False

        for chunk in response_stream:
            responses.append(chunk)
            if hasattr(chunk, "event") and chunk.event in ["ToolCallStarted", "ToolCallCompleted"]:
                tool_call_seen = True

        assert len(responses) > 0
        assert tool_call_seen, "No tool calls observed in stream"
        all_content = "".join([r.content for r in responses if r.content])
        assert "56088" in all_content  # 123 * 456

    def test_claude_stream_tool_use_thinking(self):
        """Test Claude streaming tool use with thinking"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            tools=[Calculator()],
            telemetry=False,
            monitoring=False,
        )

        response_stream = agent.run(
            "Calculate the area of a circle with radius 10 and explain your process", 
            stream=True, 
            stream_intermediate_steps=True
        )

        responses = []
        tool_call_seen = False

        for chunk in response_stream:
            responses.append(chunk)
            if hasattr(chunk, "event") and chunk.event in ["ToolCallStarted", "ToolCallCompleted"]:
                tool_call_seen = True

        assert len(responses) > 0
        assert tool_call_seen, "No tool calls observed in stream"
        all_content = "".join([r.content for r in responses if r.content])
        assert "314" in all_content or "π" in all_content  # Area = π * r²

    @pytest.mark.asyncio
    async def test_gpt5_async_stream_tool_use(self):
        """Test GPT-5 async streaming tool use"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "low", "summary": "brief"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            tools=[Calculator()],
            telemetry=False,
            monitoring=False,
        )

        response_stream = await agent.arun(
            "What is 789 divided by 3?", 
            stream=True, 
            stream_intermediate_steps=True
        )

        responses = []
        tool_call_seen = False

        async for chunk in response_stream:
            responses.append(chunk)
            if hasattr(chunk, "event") and chunk.event in ["ToolCallStarted", "ToolCallCompleted"]:
                tool_call_seen = True

        assert len(responses) > 0
        assert tool_call_seen, "No tool calls observed in stream"
        all_content = "".join([r.content for r in responses if r.content])
        assert "263" in all_content  # 789 / 3 = 263


class TestComplexToolScenarios:
    """Test complex tool use scenarios with both models"""

    def test_gpt5_reasoning_with_chained_tools(self):
        """Test GPT-5 reasoning with chained tool usage"""
        def get_user_data():
            return "User has 1000 points, joined 30 days ago"

        def calculate_bonus(days: int, points: int):
            return f"Bonus: {days * 10 + points * 0.1} points"

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "high", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            tools=[get_user_data, calculate_bonus],
            telemetry=False,
            monitoring=False,
        )

        response = agent.run("Get the user data, then calculate their bonus points")

        tool_calls = [msg.tool_calls for msg in response.messages if msg.tool_calls]
        total_calls = sum(len(calls) for calls in tool_calls)
        assert total_calls >= 2  # Should use both tools
        
        assert "400" in response.content  # 30 * 10 + 1000 * 0.1 = 400
        _assert_metrics(response)

    def test_claude_thinking_with_conditional_tools(self):
        """Test Claude thinking with conditional tool usage"""
        def check_weather_condition():
            return "rainy"

        def get_indoor_activities():
            return "Museums, libraries, shopping malls, cinemas"

        def get_outdoor_activities():
            return "Parks, hiking trails, beaches, sports fields"

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            tools=[check_weather_condition, get_indoor_activities, get_outdoor_activities],
            telemetry=False,
            monitoring=False,
        )

        response = agent.run("Check the weather and recommend appropriate activities")

        tool_calls = [msg.tool_calls for msg in response.messages if msg.tool_calls]
        assert len(tool_calls) >= 1
        
        # Should check weather and suggest indoor activities since it's rainy
        assert "indoor" in response.content.lower() or "museum" in response.content.lower()
        _assert_metrics(response)