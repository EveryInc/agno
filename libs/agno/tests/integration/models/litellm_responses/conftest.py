"""Configuration and fixtures for LiteLLMResponses integration tests."""

import os
import pytest
import requests
from typing import Dict, Any

# Test configuration
LITELLM_PROXY_URL = "http://localhost:4000"
LITELLM_API_KEY = "sk-1234"


def check_litellm_proxy() -> bool:
    """Check if LiteLLM proxy is available for testing."""
    try:
        headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
        response = requests.get(f"{LITELLM_PROXY_URL}/health", headers=headers, timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="session")
def litellm_proxy_available():
    """Check if LiteLLM proxy is available for testing."""
    available = check_litellm_proxy()
    if not available:
        pytest.skip(
            f"LiteLLM proxy not available at {LITELLM_PROXY_URL}. "
            "Please start a LiteLLM proxy server for integration tests."
        )
    return available


@pytest.fixture
def gpt5_config() -> Dict[str, Any]:
    """Configuration for GPT-5 model testing."""
    return {
        "id": "gpt-5",
        "api_base": LITELLM_PROXY_URL,
        "api_key": LITELLM_API_KEY,
        "temperature": None,
        "top_p": None,
    }


@pytest.fixture  
def claude_config() -> Dict[str, Any]:
    """Configuration for Claude Opus 4.1 model testing."""
    return {
        "id": "claude-opus-4-1",
        "api_base": LITELLM_PROXY_URL,
        "api_key": LITELLM_API_KEY,
        "temperature": 0.7,
        "top_p": None,
    }


@pytest.fixture
def gpt5_reasoning_config() -> Dict[str, Any]:
    """Configuration for GPT-5 with reasoning enabled."""
    return {
        "id": "gpt-5",
        "api_base": LITELLM_PROXY_URL,
        "api_key": LITELLM_API_KEY,
        "reasoning": {"effort": "medium", "summary": "detailed"},
        "temperature": None,
        "top_p": None,
    }


@pytest.fixture
def claude_thinking_config() -> Dict[str, Any]:
    """Configuration for Claude with thinking enabled."""
    return {
        "id": "claude-opus-4-1",
        "api_base": LITELLM_PROXY_URL,
        "api_key": LITELLM_API_KEY,
        "temperature": 1.0,
        "top_p": None,
        "request_params": {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024
            }
        }
    }


def _has_optional_dependency(package_name: str) -> bool:
    """Check if an optional dependency is available."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


# Skip marks for optional dependencies
pytest_mark_duckduckgo = pytest.mark.skipif(
    not _has_optional_dependency("duckduckgo_search"),
    reason="duckduckgo_search not installed"
)

pytest_mark_yfinance = pytest.mark.skipif(
    not _has_optional_dependency("yfinance"),
    reason="yfinance not installed"
)

pytest_mark_requests = pytest.mark.skipif(
    not _has_optional_dependency("requests"),
    reason="requests not installed"
)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "requires_proxy: mark test as requiring LiteLLM proxy server"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (takes more than a few seconds)"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically apply markers based on test requirements."""
    for item in items:
        # Mark all tests in this directory as requiring proxy
        if "litellm_responses" in str(item.fspath):
            item.add_marker(pytest.mark.requires_proxy)
        
        # Mark tests with multiple API calls as slow
        if any(keyword in item.name for keyword in ["comparison", "vs", "multiple"]):
            item.add_marker(pytest.mark.slow)