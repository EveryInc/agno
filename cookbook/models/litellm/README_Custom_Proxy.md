# LiteLLM Responses with Custom Proxy

This guide shows how to use `LiteLLMResponses` with your self-hosted LiteLLM proxy instead of the enterprise offering.

## Prerequisites

1. **LiteLLM Proxy Running**: Make sure your LiteLLM proxy is running on `localhost:4000`
2. **API Key**: Your custom API key (e.g., `sk-1234`)
3. **Model Configuration**: Models configured in your proxy's config file

## Configuration Methods

### Method 1: Direct Configuration (Recommended)

```python
from agno.agent import Agent
from agno.models.litellm import LiteLLMResponses

agent = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",  # Model name from your proxy config
        api_base="http://localhost:4000",
        api_key="sk-1234",  # Your custom API key
        temperature=0.7,
        max_output_tokens=500,
    ),
    markdown=True,
)

response = agent.print_response("Hello world!")
```

### Method 2: Environment Variables

```bash
export LITELLM_API_KEY="sk-1234"
export LITELLM_API_BASE="http://localhost:4000"
```

```python
from agno.agent import Agent
from agno.models.litellm import LiteLLMResponses

# Will automatically use environment variables
agent = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",
        temperature=0.7,
    ),
    markdown=True,
)
```

### Method 3: Bearer Token Format

If your proxy expects the Bearer format:

```python
agent = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",
        api_base="http://localhost:4000",
        api_key="Bearer sk-1234",  # Include Bearer prefix
        temperature=0.7,
    ),
    markdown=True,
)
```

## Advanced Configuration

### Custom Headers and Timeouts

```python
agent = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",
        api_base="http://localhost:4000",
        api_key="sk-1234",
        timeout=60.0,  # 60 second timeout
        max_retries=3,
        default_headers={
            "User-Agent": "MyApp/1.0",
            "X-Custom-Header": "value",
        },
    ),
    markdown=True,
)
```

### Reasoning Models with Proxy

For reasoning models (o3, o4-mini) with session continuity:

```python
reasoning_agent = Agent(
    model=LiteLLMResponses(
        id="o3-mini",  # Ensure this model is in your proxy config
        api_base="http://localhost:4000",
        api_key="sk-1234",
        reasoning_effort="medium",  # minimal, medium, high
        store=True,  # Enable session continuity
        max_output_tokens=2000,
    ),
    markdown=True,
)

# Multi-turn conversation with session continuity
reasoning_agent.print_response("Solve this math problem step by step: 2x + 5 = 13")
reasoning_agent.print_response("Now solve for x in the equation 3x - 7 = 2x + 1")
```

## LiteLLM Proxy Setup

### Sample proxy config.yaml

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: your-openai-key
      
  - model_name: o3-mini
    litellm_params:
      model: openai/o3-mini
      api_key: your-openai-key

  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: your-anthropic-key

general_settings:
  master_key: sk-1234  # Your custom API key
```

### Starting the proxy

```bash
litellm --config config.yaml --port 4000
```

## Troubleshooting

### Connection Issues

1. **Check proxy status**: Visit `http://localhost:4000/health`
2. **Verify API key**: Make sure it matches your proxy's master_key
3. **Check model availability**: Ensure the model ID exists in your proxy config

### Debug Mode

Enable debug logging to see request details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your LiteLLMResponses code here
```

### Common Error Messages

- **Connection refused**: Proxy not running on localhost:4000
- **401 Unauthorized**: API key mismatch
- **404 Model not found**: Model not configured in proxy
- **Timeout**: Increase timeout parameter or check proxy performance

## Benefits of Custom Proxy

1. **Cost Control**: Manage your own API usage and billing
2. **Model Selection**: Configure exactly the models you want
3. **Custom Logic**: Add middleware, logging, rate limiting
4. **Data Privacy**: All traffic goes through your infrastructure
5. **Load Balancing**: Distribute requests across multiple API keys
6. **Fallback Models**: Automatic failover between different providers

## Advanced Features

### Tool Usage with Custom Proxy

```python
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",
        api_base="http://localhost:4000",
        api_key="sk-1234",
    ),
    tools=[DuckDuckGoTools()],
    markdown=True,
)

agent.print_response("Search for the latest news about AI")
```

### Structured Outputs

```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    location: str
    temperature: int
    description: str

agent = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",
        api_base="http://localhost:4000",
        api_key="sk-1234",
    ),
    markdown=True,
)

response = agent.run("Get weather for San Francisco", response_model=WeatherReport)
```

## Example LiteLLM Proxy Commands

```bash
# Basic proxy
litellm --config config.yaml

# With custom port
litellm --config config.yaml --port 4000

# With debug mode
litellm --config config.yaml --debug

# With specific host binding
litellm --config config.yaml --host 0.0.0.0 --port 4000
```

This setup gives you full control over your LLM infrastructure while using the powerful agno framework!