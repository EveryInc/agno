from agno.agent import Agent
from agno.models.litellm import LiteLLMResponses

# Create an agent using the LiteLLM Responses API
agent = Agent(
    model=LiteLLMResponses(
        id="gpt-4o",  # You can use any LiteLLM-supported model here
        temperature=0.7,
        max_output_tokens=500,
    ),
    markdown=True,
)

# Simple interaction
agent.print_response("Tell me a short story about a robot learning to paint.")