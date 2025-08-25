from agno.agent import Agent
from agno.models.litellm import LiteLLMResponses

# Create an agent using a reasoning model with the LiteLLM Responses API
reasoning_agent = Agent(
    model=LiteLLMResponses(
        id="o3-mini",  # Reasoning model that supports session continuity
        reasoning_effort="medium",  # Can be "minimal", "medium", or "high"
        store=True,  # Enable storage for session continuity
        max_output_tokens=1000,
    ),
    markdown=True,
)

# Multi-turn conversation that will maintain context through previous_response_id
reasoning_agent.print_response("I need to solve this math problem: If I have 15 apples and I give away 1/3 of them, then buy twice as many as I gave away, how many apples do I have?")

# Continue the conversation - this will use previous_response_id for session continuity
reasoning_agent.print_response("Now, if I eat 3 of those apples, what percentage of my original apples do I have left?")