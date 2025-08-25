# LiteLLM Models

This directory contains LiteLLM model implementations for the agno framework.

## Available Models

- **`LiteLLM`** - Standard chat completions API implementation
- **`LiteLLMResponses`** - Advanced responses API with reasoning support
- **`LiteLLMOpenAI`** - OpenAI-compatible interface

## LiteLLMResponses (Recommended)

The `LiteLLMResponses` class uses LiteLLM's beta `/responses` API endpoint, which provides advanced features not available in the standard chat completions API.

### Supported Models

#### GPT-5
- **Model ID**: `gpt-5`
- **Special Features**: Reasoning capabilities with configurable effort levels
- **Parameters**: No temperature support (automatically handled)

#### Claude Opus 4.1
- **Model ID**: `claude-opus-4-1`
- **Special Features**: Thinking capabilities with token budgets
- **Parameters**: Temperature must be 1.0 when thinking enabled

### Key Features

- **Reasoning/Thinking Content**: Capture detailed model reasoning process
- **Session Continuity**: Chain responses with `previous_response_id`
- **Multimodal Support**: Images, audio, and file inputs
- **Structured Output**: Pydantic model responses
- **Tool Usage**: Function calling and parallel tool execution
- **Custom Proxy**: Support for self-hosted LiteLLM instances
- **Streaming**: Full streaming support for real-time responses

### Basic Usage

```python
from agno.models.litellm import LiteLLMResponses
from agno.agent import Agent

# GPT-5 with reasoning
model = LiteLLMResponses(
    id="gpt-5",
    api_base="http://localhost:4000",
    api_key="sk-1234",
    reasoning={"effort": "medium", "summary": "detailed"}
)

# Claude Opus 4.1 with thinking
model = LiteLLMResponses(
    id="claude-opus-4-1",
    api_base="http://localhost:4000", 
    api_key="sk-1234",
    temperature=1.0,
    request_params={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1024
        }
    }
)

agent = Agent(model=model)
response = agent.run("Explain quantum computing")
```

### Configuration Options

#### Custom Proxy Setup
```python
model = LiteLLMResponses(
    id="gpt-5",
    api_base="https://your-litellm-proxy.com",
    api_key="your-api-key",
    custom_headers={"Custom-Header": "value"},
    timeout=60.0
)
```

#### Session Continuity
```python
# Enable automatic session chaining
agent = Agent(
    model=LiteLLMResponses(
        id="gpt-5",
        store=True,  # Enable response storage
        # ... other params
    ),
    add_history_to_messages=True
)
```

### Model-Specific Parameters

#### GPT-5 Parameters
```python
LiteLLMResponses(
    id="gpt-5",
    reasoning={
        "effort": "low" | "medium" | "high",
        "summary": "minimal" | "detailed"
    },
    temperature=None,  # Not supported, automatically set
    top_p=None,        # Use with caution
)
```

#### Claude Opus 4.1 Parameters
```python
LiteLLMResponses(
    id="claude-opus-4-1",
    temperature=1.0,   # Required when thinking enabled
    top_p=None,        # Don't use with temperature
    request_params={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 512 | 1024 | 2048
        }
    }
)
```

### Advanced Features

#### Structured Output
```python
from pydantic import BaseModel, Field

class Response(BaseModel):
    summary: str = Field(..., description="Brief summary")
    details: list[str] = Field(..., description="Detailed points")

agent = Agent(
    model=LiteLLMResponses(id="gpt-5", ...),
    response_model=Response
)
```

#### Tool Usage
```python
from agno.tools.duckduckgo import DuckDuckGo

agent = Agent(
    model=LiteLLMResponses(id="claude-opus-4-1", ...),
    tools=[DuckDuckGo()]
)
```

#### Multimodal Input
```python
from agno.models.message import Message

message = Message(
    role="user",
    content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "image": "/path/to/image.jpg"}
    ]
)

response = agent.run(message)
```

## Testing

Run the comprehensive test suite:

```bash
# All LiteLLMResponses tests
pytest libs/agno/tests/integration/models/litellm_responses/

# Specific test categories
pytest libs/agno/tests/integration/models/litellm_responses/test_basic.py
pytest libs/agno/tests/integration/models/litellm_responses/test_reasoning_content.py
```

**Note**: Integration tests require a running LiteLLM proxy server on `localhost:4000` with API key `sk-1234`.

## Examples

See the cookbook for complete examples:
- [Custom Proxy Configuration](../../../../cookbook/models/litellm/responses_custom_proxy.py)

## Troubleshooting

### Common Issues

1. **Temperature conflicts with Claude thinking**
   - Solution: Set `temperature=1.0` and `top_p=None` when using thinking

2. **GPT-5 temperature parameter ignored**
   - Expected: GPT-5 doesn't support temperature, automatically handled

3. **Reasoning content not captured**
   - Check model configuration and ensure reasoning/thinking is properly enabled

4. **Connection errors to proxy**
   - Verify proxy is running and API key is correct
   - Check `api_base` URL format

For more help, see the [agno documentation](https://docs.agno.com) or join our [Discord](https://discord.gg/4MtYHHrgA8).