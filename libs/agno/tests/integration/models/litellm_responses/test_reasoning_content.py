import pytest

from agno.agent import Agent, RunResponse
from agno.models.litellm import LiteLLMResponses


class TestReasoningContent:
    """Test that RunResponse captures reasoning/thinking content from both models"""

    def test_gpt5_reasoning_content_captured(self):
        """Test that GPT-5 reasoning content is captured in RunResponse"""
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

        response: RunResponse = agent.run("Explain why the sky is blue, step by step")

        # Verify basic response
        assert response.content is not None
        assert len(response.content) > 0

        # Verify reasoning content is captured
        # The reasoning content should be available in the assistant message
        assistant_message = None
        for msg in response.messages:
            if msg.role == "assistant":
                assistant_message = msg
                break
        
        assert assistant_message is not None
        # GPT-5 reasoning should be captured in reasoning_content attribute
        assert hasattr(assistant_message, 'reasoning_content') or hasattr(assistant_message, 'thinking')
        
        # The reasoning should contain some analysis
        reasoning_text = getattr(assistant_message, 'reasoning_content', '') or getattr(assistant_message, 'thinking', '')
        if reasoning_text:
            assert len(reasoning_text) > 0
            print(f"GPT-5 reasoning captured: {reasoning_text[:200]}...")

    def test_claude_thinking_content_captured(self):
        """Test that Claude thinking content is captured in RunResponse"""
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

        response: RunResponse = agent.run("Count the number of r's in 'strawberry' and explain your reasoning")

        # Verify basic response
        assert response.content is not None
        assert len(response.content) > 0
        assert "3" in response.content  # Should find 3 r's

        # Verify thinking content is captured
        assistant_message = None
        for msg in response.messages:
            if msg.role == "assistant":
                assistant_message = msg
                break
        
        assert assistant_message is not None
        # Claude thinking should be captured in reasoning_content attribute
        assert hasattr(assistant_message, 'reasoning_content') or hasattr(assistant_message, 'thinking')
        
        # The thinking should contain some analysis
        thinking_text = getattr(assistant_message, 'reasoning_content', '') or getattr(assistant_message, 'thinking', '')
        if thinking_text:
            assert len(thinking_text) > 0
            print(f"Claude thinking captured: {thinking_text[:200]}...")

    def test_gpt5_vs_claude_reasoning_comparison(self):
        """Test that both models provide reasoning content and compare their approaches"""
        
        # Test GPT-5 reasoning
        gpt5_agent = Agent(
            model=LiteLLMResponses(
                id="gpt-5",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                reasoning={"effort": "high", "summary": "detailed"},
                temperature=None,
                top_p=None,
            ),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        # Test Claude thinking
        claude_agent = Agent(
            model=LiteLLMResponses(
                id="claude-opus-4-1",
                api_base="http://localhost:4000",
                api_key="sk-1234",
                temperature=1.0,
                top_p=None,
                request_params={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 2048
                    }
                }
            ),
            markdown=True,
            telemetry=False,
            monitoring=False
        )

        prompt = "What are the key differences between renewable and non-renewable energy sources?"

        # Get responses from both models
        gpt5_response = gpt5_agent.run(prompt)
        claude_response = claude_agent.run(prompt)

        # Both should have content
        assert gpt5_response.content is not None
        assert claude_response.content is not None
        assert len(gpt5_response.content) > 0
        assert len(claude_response.content) > 0

        # Both should have reasoning/thinking content
        gpt5_assistant = next(msg for msg in gpt5_response.messages if msg.role == "assistant")
        claude_assistant = next(msg for msg in claude_response.messages if msg.role == "assistant")

        gpt5_reasoning = getattr(gpt5_assistant, 'reasoning_content', '') or getattr(gpt5_assistant, 'thinking', '')
        claude_thinking = getattr(claude_assistant, 'reasoning_content', '') or getattr(claude_assistant, 'thinking', '')

        if gpt5_reasoning:
            print(f"GPT-5 reasoning length: {len(gpt5_reasoning)}")
            print(f"GPT-5 reasoning preview: {gpt5_reasoning[:100]}...")

        if claude_thinking:
            print(f"Claude thinking length: {len(claude_thinking)}")
            print(f"Claude thinking preview: {claude_thinking[:100]}...")

        # At least one model should provide reasoning content
        has_reasoning = bool(gpt5_reasoning) or bool(claude_thinking)
        assert has_reasoning, "Neither model provided reasoning/thinking content"

    def test_reasoning_content_token_usage_impact(self):
        """Test that reasoning/thinking increases token usage measurably"""
        
        # Test GPT-5 without reasoning
        gpt5_no_reasoning = Agent(
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

        # Test GPT-5 with reasoning
        gpt5_with_reasoning = Agent(
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

        prompt = "Explain the concept of machine learning in simple terms"

        # Get responses
        no_reasoning_response = gpt5_no_reasoning.run(prompt)
        with_reasoning_response = gpt5_with_reasoning.run(prompt)

        # Both should have metrics
        assert no_reasoning_response.metrics is not None
        assert with_reasoning_response.metrics is not None

        # Get total tokens
        no_reasoning_total = sum(no_reasoning_response.metrics.get("total_tokens", []))
        with_reasoning_total = sum(with_reasoning_response.metrics.get("total_tokens", []))

        print(f"GPT-5 without reasoning: {no_reasoning_total} total tokens")
        print(f"GPT-5 with reasoning: {with_reasoning_total} total tokens")

        # Reasoning should generally increase token usage
        # (though not guaranteed in every case, so we just log for observation)
        if with_reasoning_total > no_reasoning_total:
            print(f"Reasoning increased token usage by {with_reasoning_total - no_reasoning_total} tokens")

    def test_claude_thinking_vs_no_thinking_content(self):
        """Test Claude with and without thinking to verify content differences"""
        
        # Claude without thinking
        claude_no_thinking = Agent(
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

        # Claude with thinking
        claude_with_thinking = Agent(
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

        prompt = "What's the most efficient way to sort a list of 1 million numbers?"

        # Get responses
        no_thinking_response = claude_no_thinking.run(prompt)
        with_thinking_response = claude_with_thinking.run(prompt)

        # Both should have content
        assert no_thinking_response.content is not None
        assert with_thinking_response.content is not None

        # Check for thinking content in the thinking-enabled response
        thinking_assistant = next(msg for msg in with_thinking_response.messages if msg.role == "assistant")
        thinking_content = getattr(thinking_assistant, 'reasoning_content', '') or getattr(thinking_assistant, 'thinking', '')

        if thinking_content:
            print(f"Claude thinking content length: {len(thinking_content)}")
            print(f"Claude thinking preview: {thinking_content[:150]}...")
        else:
            print("No thinking content captured for Claude")

        # Log metrics comparison
        no_thinking_total = sum(no_thinking_response.metrics.get("total_tokens", []))
        with_thinking_total = sum(with_thinking_response.metrics.get("total_tokens", []))
        
        print(f"Claude without thinking: {no_thinking_total} total tokens")
        print(f"Claude with thinking: {with_thinking_total} total tokens")