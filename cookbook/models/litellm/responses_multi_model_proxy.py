from agno.agent import Agent
from agno.models.litellm import LiteLLMResponses

# Your proxy configuration
PROXY_BASE_URL = "http://localhost:4000"
PROXY_API_KEY = "sk-1234"

print("🚀 Multi-Model LiteLLM Proxy Examples")
print(f"Proxy: {PROXY_BASE_URL}")
print(f"API Key: {PROXY_API_KEY[:8]}...")

# Example 1: OpenAI GPT-4o
print("\n=== OpenAI GPT-4o ===")
gpt4_agent = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",  # As configured in your proxy
        api_base=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
        temperature=0.7,
        max_output_tokens=1000,
    ),
    markdown=True,
)

# Example 2: OpenAI o3-mini (reasoning model)
print("\n=== OpenAI o3-mini (Reasoning) ===")
o3_agent = Agent(
    model=LiteLLMResponses(
        id="o3-mini",  # Make sure this is in your proxy config
        api_base=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
        reasoning_effort="medium",
        store=True,  # Enable session continuity for reasoning
        max_output_tokens=2000,
        temperature=0.3,
    ),
    markdown=True,
)

# Example 3: Claude (if configured in your proxy)
print("\n=== Anthropic Claude ===")
claude_agent = Agent(
    model=LiteLLMResponses(
        id="claude-3-5-sonnet",  # As configured in your proxy
        api_base=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
        temperature=0.8,
        max_output_tokens=1500,
    ),
    markdown=True,
)

# Example 4: Gemini (if configured in your proxy)
print("\n=== Google Gemini ===")
gemini_agent = Agent(
    model=LiteLLMResponses(
        id="gemini-1.5-pro",  # As configured in your proxy
        api_base=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
        temperature=0.6,
        max_output_tokens=1200,
    ),
    markdown=True,
)

# Example 5: Local model (if you have local models in proxy)
print("\n=== Local Model ===")
local_agent = Agent(
    model=LiteLLMResponses(
        id="local-llama-3-8b",  # Your local model name
        api_base=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
        temperature=0.9,
        max_output_tokens=800,
    ),
    markdown=True,
)

# Example proxy config.yaml that would support these models:
proxy_config_example = """
# Example config.yaml for your LiteLLM proxy

model_list:
  # OpenAI Models
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: o3-mini
    litellm_params:
      model: openai/o3-mini
      api_key: os.environ/OPENAI_API_KEY

  # Anthropic Models
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  # Google Models
  - model_name: gemini-1.5-pro
    litellm_params:
      model: gemini/gemini-1.5-pro
      api_key: os.environ/GOOGLE_API_KEY

  # Local Models (if using Ollama/vLLM/etc)
  - model_name: local-llama-3-8b
    litellm_params:
      model: ollama/llama3:8b
      api_base: http://localhost:11434

general_settings:
  master_key: sk-1234  # Your API key
  database_url: "postgresql://user:pass@localhost/litellm"  # Optional: for persistence
"""

print(f"\n📝 Example proxy configuration:")
print("Save this as config.yaml and run: litellm --config config.yaml --port 4000")
print("=" * 60)
print(proxy_config_example)

# Test function to verify model availability
def test_model_availability():
    """Test which models are available in your proxy"""
    print("\n🔍 Testing model availability...")
    
    models_to_test = [
        "gpt-4o",
        "o3-mini", 
        "claude-3-5-sonnet",
        "gemini-1.5-pro",
        "local-llama-3-8b"
    ]
    
    for model_id in models_to_test:
        try:
            test_model = LiteLLMResponses(
                id=model_id,
                api_base=PROXY_BASE_URL,
                api_key=PROXY_API_KEY,
            )
            print(f"  ✅ {model_id}: Configuration ready")
        except Exception as e:
            print(f"  ❌ {model_id}: {e}")

# Uncomment to test model availability
# test_model_availability()

print("\n🎯 Usage Examples:")
print("1. Choose the model that matches your proxy configuration")
print("2. Make sure your LiteLLM proxy is running on localhost:4000") 
print("3. Verify your API key (sk-1234) is set as master_key in proxy config")
print("4. Test with: agent.print_response('Hello world!')")

# Example usage (commented out to avoid actual API calls):
"""
# Test GPT-4o
print("\\n🧪 Testing GPT-4o...")
try:
    response = gpt4_agent.print_response("Explain quantum computing in simple terms")
    print("✅ GPT-4o is working!")
except Exception as e:
    print(f"❌ GPT-4o failed: {e}")

# Test reasoning model
print("\\n🧪 Testing o3-mini reasoning...")
try:
    response = o3_agent.print_response("Solve step by step: If a train travels 120km in 2 hours, what's its speed?")
    print("✅ o3-mini reasoning is working!")
except Exception as e:
    print(f"❌ o3-mini failed: {e}")
"""