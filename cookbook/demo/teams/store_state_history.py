#!/usr/bin/env python
"""
Demo script for testing session state history tracking.

This script demonstrates how to:
1. Create a team with session state history enabled
2. Run the team multiple times with different inputs that update the session state
3. Retrieve and display the session state history
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
import asyncio

from sqlalchemy import create_engine

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage
from agno.team import Team

# Configure PostgreSQL storage
DB_URL = os.environ.get("AGNO_DB_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
db_engine = create_engine(DB_URL)
storage = PostgresStorage(
    table_name="team_sessions_demo",
    schema="agno",
    db_engine=db_engine,
    mode="team",
)

# Create the tables if they don't exist
storage.create()

# Create a model for the team
model = OpenAIChat(id="gpt-4.1-mini")

def increase_counter(agent: Agent) -> str:
    agent.team_session_state["counter"] += 1
    return "Counter increased to " + str(agent.team_session_state["counter"])

def decrease_counter(agent: Agent) -> str:
    agent.team_session_state["counter"] -= 1
    return "Counter decreased to " + str(agent.team_session_state["counter"])

# Create a simple agent
agent = Agent(
    name="Counter",
    model=model,
    tools=[
        increase_counter,
        decrease_counter,
    ],
    description="An agent that helps count and track numbers.",
)

# Create a team with session state history enabled
team = Team(
    members=[agent],
    name="State History Demo Team",
    model=model,
    description="A team that demonstrates session state history tracking.",
    mode="coordinate",
    # Enable session state history tracking
    enable_session_state_history=True,
    # Set initial session state
    session_state={"counter": 0, "history": []},
    # Add state in messages so the agent can see and update it
    add_state_in_messages=True,
    # Configure storage
    storage=storage,
    # Add instructions for the agent
    instructions="""You are part of a team that demonstrates session state tracking. 
    The session state contains a counter and a history list.
    When the user asks you to increment the counter, add 1 to the counter and add an entry to the history list with the current timestamp.
    When the user asks you to decrement the counter, subtract 1 from the counter and add an entry to the history list with the current timestamp.
    Always respond with the current counter value and the last 3 history entries.
    """,
)

# Function to run the team and print the response
async def run_team(message: str) -> None:
    print(f"\n=== Running team with message: '{message}' ===")
    await team.aprint_response(message)
    # print(f"Current session state: {json.dumps(team.session_state, indent=2)}\n")

# Function to print session state history
def print_session_state_history() -> None:
    print("\n=== Session State History ===")
    history = team.get_session_state_history()
    
    if not history:
        print("No session state history found.")
        return
        
    for entry in history:
        run_id = entry["run_id"]
        state = entry["state"]
        created_at = datetime.fromtimestamp(entry["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Run ID: {run_id}")
        print(f"Created at: {created_at}")
        print(f"State: {json.dumps(state, indent=2)}")
        print("---")

# Main demo
async def main():
    # Run the team multiple times with different inputs
    await run_team("Hello! Let's start tracking some state.")
    await run_team("Increment the counter please.")
    await run_team("Increment it again!")
    await run_team("Now decrement the counter.")
    await run_team("Increment one more time.")
    
    # Print the session state history
    print_session_state_history()

if __name__ == "__main__":
    asyncio.run(main())
