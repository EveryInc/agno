from agno.agent import Agent
from agno.models.litellm import LiteLLMResponses
import os

# Method 1: Direct configuration in the model
print("=== Method 1: Direct Configuration ===")
agent_direct = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",  # Your model name as configured in the proxy
        api_base="http://localhost:4000",
        api_key="sk-1234",  # Your custom API key
        temperature=0.7,
        max_output_tokens=500,
    ),
    markdown=True,
)

# Method 2: Using environment variables
print("\n=== Method 2: Environment Variables ===")
os.environ["LITELLM_API_KEY"] = "sk-1234"
os.environ["LITELLM_API_BASE"] = "http://localhost:4000"

agent_env = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",
        temperature=0.7,
    ),
    markdown=True,
)

# Method 3: Custom headers and timeout for your proxy
print("\n=== Method 3: Advanced Proxy Configuration ===")
agent_advanced = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",
        api_base="http://localhost:4000",
        api_key="sk-1234",
        timeout=60.0,  # Custom timeout for your proxy
        max_retries=3,
        default_headers={
            "User-Agent": "Agno-LiteLLM/1.0",
            "X-Custom-Header": "MyApp",  # Any custom headers your proxy needs
        },
        temperature=0.7,
        max_output_tokens=1000,
    ),
    markdown=True,
)

# Method 4: Reasoning model with custom proxy
print("\n=== Method 4: Reasoning Model with Custom Proxy ===")
reasoning_agent = Agent(
    model=LiteLLMResponses(
        id="o3-mini",  # Make sure this model is available in your proxy config
        api_base="http://localhost:4000",
        api_key="sk-1234",
        reasoning_effort="medium",
        store=True,  # Enable storage for session continuity
        max_output_tokens=2000,
        temperature=0.3,
    ),
    markdown=True,
)

# Method 5: Using Bearer token format (if your proxy expects it)
print("\n=== Method 5: Bearer Token Format ===")
agent_bearer = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",
        api_base="http://localhost:4000",
        api_key="Bearer sk-1234",  # Include Bearer prefix if needed
        temperature=0.7,
    ),
    markdown=True,
)

print("\n🚀 All proxy configurations are ready!")
print("Choose the method that works best with your LiteLLM proxy setup.")

# Example usage (commented out to avoid actual API calls)
"""
# Test the connection
try:
    response = agent_direct.print_response("Hello! Can you tell me about the weather?")
    print("✅ Proxy connection successful!")
except Exception as e:
    print(f"❌ Proxy connection failed: {e}")
    print("Make sure your LiteLLM proxy is running on localhost:4000")
"""