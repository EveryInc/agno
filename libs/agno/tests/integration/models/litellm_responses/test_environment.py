"""Test environment configuration and fixtures."""

import pytest
from agno.models.litellm import LiteLLMResponses
from agno.agent import Agent


class TestEnvironmentSetup:
    """Test that the test environment is properly configured."""

    def test_litellm_proxy_available(self, litellm_proxy_available):
        """Test that LiteLLM proxy is available."""
        # This test will be skipped if proxy is not available
        assert litellm_proxy_available is True

    def test_gpt5_config_fixture(self, gpt5_config):
        """Test GPT-5 configuration fixture."""
        assert gpt5_config["id"] == "gpt-5"
        assert gpt5_config["api_base"] == "http://localhost:4000"
        assert gpt5_config["api_key"] == "sk-1234"
        
        # Test that config can be used to create a model
        model = LiteLLMResponses(**gpt5_config)
        assert model.id == "gpt-5"

    def test_claude_config_fixture(self, claude_config):
        """Test Claude configuration fixture."""
        assert claude_config["id"] == "claude-opus-4-1"
        assert claude_config["api_base"] == "http://localhost:4000"
        assert claude_config["api_key"] == "sk-1234"
        
        # Test that config can be used to create a model
        model = LiteLLMResponses(**claude_config)
        assert model.id == "claude-opus-4-1"

    def test_parameter_validation(self, gpt5_config, claude_thinking_config):
        """Test that parameter validation works correctly."""
        
        # Test GPT-5 temperature validation
        gpt5_config_with_temp = gpt5_config.copy()
        gpt5_config_with_temp["temperature"] = 0.7
        model = LiteLLMResponses(**gpt5_config_with_temp)
        # Temperature should be set to None for GPT-5
        assert model.temperature is None
        
        # Test Claude thinking validation
        model = LiteLLMResponses(**claude_thinking_config)
        # Temperature should be 1.0 for Claude with thinking
        assert model.temperature == 1.0
        # top_p should be None
        assert model.top_p is None

    @pytest.mark.skipif(True, reason="Simple smoke test - enable to test basic connectivity")
    def test_basic_connectivity(self, litellm_proxy_available, gpt5_config):
        """Simple connectivity test (disabled by default to avoid API calls)."""
        agent = Agent(
            model=LiteLLMResponses(**gpt5_config),
            telemetry=False,
            monitoring=False
        )
        
        response = agent.run("Say 'hello' in one word")
        assert response.content is not None
        assert len(response.content) > 0