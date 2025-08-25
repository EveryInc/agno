import pytest
from pydantic import BaseModel, Field

from agno.agent import Agent, RunResponse
from agno.models.litellm import LiteLLMResponses
from agno.storage.sqlite import SqliteStorage


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


class TestBasicGPT5:
    """Test basic functionality with GPT-5 model"""

    def test_gpt5_basic_no_reasoning(self, litellm_proxy_available, gpt5_config):
        """Test GPT-5 basic functionality without reasoning"""
        agent = Agent(
            model=LiteLLMResponses(**gpt5_config),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        response: RunResponse = agent.run("Share a 2 sentence horror story")

        assert response.content is not None
        assert len(response.messages) == 3
        assert [m.role for m in response.messages] == ["system", "user", "assistant"]
        _assert_metrics(response)

    def test_gpt5_basic_with_reasoning(self, litellm_proxy_available, gpt5_reasoning_config):
        """Test GPT-5 basic functionality with reasoning enabled"""
        agent = Agent(
            model=LiteLLMResponses(**gpt5_reasoning_config),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        response: RunResponse = agent.run("Solve this step by step: What is 15 * 23?")

        assert response.content is not None
        assert len(response.messages) == 3
        assert [m.role for m in response.messages] == ["system", "user", "assistant"]
        
        # Check if reasoning content is captured
        assistant_message = response.messages[-1]
        # GPT-5 should have reasoning content when reasoning is enabled
        assert hasattr(assistant_message, 'reasoning_content') or hasattr(assistant_message, 'thinking')
        
        _assert_metrics(response)

    def test_gpt5_reasoning_effort_levels(self):
        """Test GPT-5 with different reasoning effort levels"""
        for effort in ["low", "medium", "high"]:
            agent = Agent(
                model=LiteLLMResponses(
                    id="gpt-5",
                    api_base="http://localhost:4000",
                    api_key="sk-1234",
                    reasoning={"effort": effort, "summary": "detailed"},
                    temperature=None,
                    top_p=None,
                ),
                markdown=True,
                telemetry=False,
                monitoring=False
            )

            response: RunResponse = agent.run(f"Count the number of r's in 'strawberry' (effort: {effort})")

            assert response.content is not None
            assert "3" in response.content  # Should find 3 r's
            _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_gpt5_async_basic(self):
        """Test GPT-5 async functionality without reasoning"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        response = await agent.arun("Share a 2 sentence horror story")

        assert response.content is not None
        assert len(response.messages) == 3
        assert [m.role for m in response.messages] == ["system", "user", "assistant"]
        _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_gpt5_async_with_reasoning(self):
        """Test GPT-5 async functionality with reasoning"""
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
            telemetry=False,
            monitoring=False
        )

        response = await agent.arun("Solve this step by step: What is the square root of 144?")

        assert response.content is not None
        assert "12" in response.content
        _assert_metrics(response)

    def test_gpt5_stream_no_reasoning(self):
        """Test GPT-5 streaming functionality without reasoning"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        response_stream = agent.run("Share a 2 sentence horror story", stream=True)

        # Verify it's an iterator
        assert hasattr(response_stream, "__iter__")

        responses = list(response_stream)
        assert len(responses) > 0
        for response in responses:
            assert response.content is not None

        _assert_metrics(agent.run_response)


class TestBasicClaude:
    """Test basic functionality with Claude Opus 4.1 model"""

    def test_claude_basic_no_thinking(self):
        """Test Claude basic functionality without thinking"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=0.7,  # Normal temperature when thinking is disabled
                top_p=None,  # Don't use top_p with temperature for Claude
            ),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        response: RunResponse = agent.run("Share a 2 sentence horror story")

        assert response.content is not None
        assert len(response.messages) == 3
        assert [m.role for m in response.messages] == ["system", "user", "assistant"]
        _assert_metrics(response)

    def test_claude_basic_with_thinking(self):
        """Test Claude basic functionality with thinking enabled"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,  # Required to be 1.0 when thinking is enabled
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        response: RunResponse = agent.run("Explain the process of photosynthesis step by step")

        assert response.content is not None
        assert len(response.messages) == 3
        assert [m.role for m in response.messages] == ["system", "user", "assistant"]
        _assert_metrics(response)

    def test_claude_thinking_budget_levels(self):
        """Test Claude with different thinking budget token levels"""
        for budget in [512, 1024, 2048]:
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
                            "budget_tokens": budget
                        }
                    }
                ),
                markdown=True,
                telemetry=False,
                monitoring=False
            )

            response: RunResponse = agent.run(f"Analyze the pros and cons of renewable energy (budget: {budget})")

            assert response.content is not None
            assert "renewable" in response.content.lower()
            _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_claude_async_basic(self):
        """Test Claude async functionality without thinking"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=0.7,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        response = await agent.arun("Share a 2 sentence horror story")

        assert response.content is not None
        assert len(response.messages) == 3
        assert [m.role for m in response.messages] == ["system", "user", "assistant"]
        _assert_metrics(response)

    @pytest.mark.asyncio
    async def test_claude_async_with_thinking(self):
        """Test Claude async functionality with thinking"""
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
            telemetry=False,
            monitoring=False
        )

        response = await agent.arun("Explain quantum entanglement in simple terms")

        assert response.content is not None
        assert "quantum" in response.content.lower()
        _assert_metrics(response)

    def test_claude_stream_no_thinking(self):
        """Test Claude streaming functionality without thinking"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=0.7,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        response_stream = agent.run("Share a 2 sentence horror story", stream=True)

        # Verify it's an iterator
        assert hasattr(response_stream, "__iter__")

        responses = list(response_stream)
        assert len(responses) > 0
        for response in responses:
            assert response.content is not None

        _assert_metrics(agent.run_response)


class TestMemoryAndHistory:
    """Test memory and history functionality with both models"""

    def test_gpt5_with_memory_reasoning(self):
        """Test GPT-5 with memory and reasoning"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "medium", "summary": "detailed"},
                store=True,  # Enable storage for session continuity
                temperature=None,
                top_p=None,
            ),
            add_history_to_messages=True,
            num_history_responses=5,
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        # First interaction
        response1 = agent.run("My name is Alice and I'm 25 years old")
        assert response1.content is not None

        # Second interaction should remember the information
        response2 = agent.run("What's my name and age?")
        assert "Alice" in response2.content
        assert "25" in response2.content

        # Verify session continuity through previous_response_id
        messages = agent.get_messages_for_session()
        assert len(messages) == 5
        _assert_metrics(response2)

    def test_claude_with_memory_thinking(self):
        """Test Claude with memory and thinking"""
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
            add_history_to_messages=True,
            num_history_responses=5,
            markdown=True,
            telemetry=False,
            monitoring=False,
        )

        # First interaction
        response1 = agent.run("My favorite color is blue and I'm from Canada")
        assert response1.content is not None

        # Second interaction should remember the information
        response2 = agent.run("What's my favorite color and where am I from?")
        assert "blue" in response2.content.lower()
        assert "canada" in response2.content.lower()

        # Verify messages were created
        messages = agent.get_messages_for_session()
        assert len(messages) == 5
        _assert_metrics(response2)


class TestStructuredOutput:
    """Test structured output functionality with both models"""

    def test_gpt5_response_model_no_reasoning(self):
        """Test GPT-5 structured output without reasoning"""
        class MovieScript(BaseModel):
            title: str = Field(..., description="Movie title")
            genre: str = Field(..., description="Movie genre")
            plot: str = Field(..., description="Brief plot summary")

        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
            response_model=MovieScript,
        )

        response = agent.run("Create a movie about time travel")

        # Verify structured output
        assert isinstance(response.content, MovieScript)
        assert response.content.title is not None
        assert response.content.genre is not None
        assert response.content.plot is not None

    def test_claude_response_model_no_thinking(self):
        """Test Claude structured output without thinking"""
        class Recipe(BaseModel):
            name: str = Field(..., description="Recipe name")
            ingredients: list[str] = Field(..., description="List of ingredients")
            instructions: str = Field(..., description="Cooking instructions")

        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=0.7,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False,
            response_model=Recipe,
        )

        response = agent.run("Create a simple pasta recipe")

        # Verify structured output
        assert isinstance(response.content, Recipe)
        assert response.content.name is not None
        assert len(response.content.ingredients) > 0
        assert response.content.instructions is not None


class TestStorage:
    """Test storage functionality with both models"""

    def test_gpt5_with_storage(self):
        """Test GPT-5 with SQLite storage"""
        agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=None,
                top_p=None,
            ),
            storage=SqliteStorage(table_name="gpt5_test_sessions", db_file="tmp/gpt5_test.db"),
            add_history_to_messages=True,
            telemetry=False,
            monitoring=False,
        )
        
        agent.run("Hello")
        assert len(agent.run_response.messages) == 2
        agent.run("Hello 2")
        assert len(agent.run_response.messages) == 4
        agent.run("Hello 3")
        assert len(agent.run_response.messages) == 6

    def test_claude_with_storage(self):
        """Test Claude with SQLite storage"""
        agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=0.7,
                top_p=None,
            ),
            storage=SqliteStorage(table_name="claude_test_sessions", db_file="tmp/claude_test.db"),
            add_history_to_messages=True,
            telemetry=False,
            monitoring=False,
        )
        
        agent.run("Hello")
        assert len(agent.run_response.messages) == 2
        agent.run("Hello 2")
        assert len(agent.run_response.messages) == 4
        agent.run("Hello 3")
        assert len(agent.run_response.messages) == 6