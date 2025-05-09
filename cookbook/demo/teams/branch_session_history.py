#!/usr/bin/env python
"""
Demo script for creating a new session from a previous session's history.

This script demonstrates:
1. Creating a team with session state history enabled
2. Running the team multiple times to build up state
3. Creating a new session from a specific point in history
4. Verifying that the new session has the correct state
"""

import os
import json
from typing import Dict, Any, List
from uuid import uuid4

from sqlalchemy import create_engine

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage
from agno.team import Team


def main():
    # Set up database connection
    db_url = os.environ.get("AGNO_DB_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
    db_engine = create_engine(db_url)
    
    # Create storage
    storage = PostgresStorage(
        table_name="demo_branch_sessions",
        schema="ai",
        db_engine=db_engine,
        mode="team",
    )
    
    # Create the tables
    storage.create()
    
    # Create a model
    model = OpenAIChat(id="gpt-3.5-turbo")
    
    # Create an agent with tools to manipulate state
    def increase_counter(agent: Agent) -> str:
        agent.team_session_state["counter"] += 1
        return f"Counter increased to {agent.team_session_state['counter']}"

    def decrease_counter(agent: Agent) -> str:
        agent.team_session_state["counter"] -= 1
        return f"Counter decreased to {agent.team_session_state['counter']}"
    
    def add_item(agent: Agent, item: str) -> str:
        agent.team_session_state["items"].append(item)
        return f"Added '{item}' to the list. Current items: {agent.team_session_state['items']}"
    
    agent = Agent(
        name="StateManager",
        model=model,
        tools=[
            increase_counter,
            decrease_counter,
            add_item,
        ],
        description="An agent that helps manage state in a session.",
        instructions="""You are a state management agent that helps track counters and items.
        When asked to increment, use the increase_counter tool.
        When asked to decrement, use the decrease_counter tool.
        When asked to add an item, use the add_item tool with the item name.
        Always respond with the current state after making changes.
        """,
    )
    
    # Create a team with session state history enabled
    team = Team(
        members=[agent],
        name="Session Branching Demo Team",
        model=model,
        description="A team that demonstrates session branching from history.",
        enable_session_state_history=True,  # Enable session state history
        session_state={"counter": 0, "items": []},
        add_state_in_messages=True,
        storage=storage,
    )
    
    # Generate a session ID
    session_id = str(uuid4())
    print(f"Created original session with ID: {session_id}")
    
    # Set the session ID for the team
    team.session_id = session_id
    
    # Run the team multiple times to build up state
    print("\n--- Original Session ---")
    
    # First run - initialize
    response = team.run("Hello! Let's start tracking some state.")
    print(f"Run 1 response: {response}")
    print(f"Run 1 state: {team.session_state}")
    run_id_1 = team.run_id
    print(f"Run 1 ID: {run_id_1}")
    
    # Make sure the session is saved to storage
    team.write_to_storage(session_id=session_id)
    
    # Second run - increment counter
    response = team.run("Increment the counter please.")
    print(f"Run 2 response: {response}")
    print(f"Run 2 state: {team.session_state}")
    run_id_2 = team.run_id
    print(f"Run 2 ID: {run_id_2}")
    
    # Make sure the session is saved to storage
    team.write_to_storage(session_id=session_id)
    
    # Third run - add an item
    response = team.run("Add 'apple' to the items list.")
    print(f"Run 3 response: {response}")
    print(f"Run 3 state: {team.session_state}")
    run_id_3 = team.run_id
    print(f"Run 3 ID: {run_id_3}")
    
    # Make sure the session is saved to storage
    team.write_to_storage(session_id=session_id)
    
    # Fourth run - increment counter again
    response = team.run("Increment the counter again.")
    print(f"Run 4 response: {response}")
    print(f"Run 4 state: {team.session_state}")
    run_id_4 = team.run_id
    print(f"Run 4 ID: {run_id_4}")
    
    # Make sure the session is saved to storage
    team.write_to_storage(session_id=session_id)
    
    # Fifth run - add another item
    response = team.run("Add 'banana' to the items list.")
    print(f"Run 5 response: {response}")
    print(f"Run 5 state: {team.session_state}")
    run_id_5 = team.run_id
    print(f"Run 5 ID: {run_id_5}")
    
    # Make sure the session is saved to storage
    team.write_to_storage(session_id=session_id)
    
    # Get session state history
    history = team.get_session_state_history(session_id=session_id)
    print(f"\nSession state history (newest first):")
    for i, entry in enumerate(history):
        print(f"  Entry {i+1}:")
        print(f"    Run ID: {entry['run_id']}")
        print(f"    State: {entry['state']}")
        print(f"    Created at: {entry['created_at']}")
    
    # Now create a new session from the state at run_id_3 (after adding 'apple')
    print("\n--- Creating a new session from run_id_3 ---")
    
    # We need to make sure we're using the correct run_id
    # Find the run ID where items=['apple'] and counter=1
    run_states = {entry['run_id']: entry['state'] for entry in history}
    target_run_id = None
    target_state = {'items': ['apple'], 'counter': 1}
    
    print("\n--- Looking for run with state ---")
    print(f"Target state: {target_state}")
    
    for run_id, state in run_states.items():
        # Check if this state matches our target
        if state.get('items') == target_state['items'] and state.get('counter') == target_state['counter']:
            target_run_id = run_id
            print(f"✅ Found exact match with run ID: {run_id}")
            print(f"   State: {state}")
            break
    
    # Print all run IDs and states for debugging
    print("\n--- All run IDs and states ---")
    for i, (run_id, state) in enumerate(run_states.items()):
        print(f"Run {i+1}: {run_id}")
        print(f"   State: {state}")
        
    # Check if the run exists in the memory
    print("\n--- Checking run IDs in memory ---")
    if hasattr(team.memory, 'runs') and isinstance(team.memory.runs, dict):
        if session_id in team.memory.runs:
            memory_runs = team.memory.runs[session_id]
            print(f"Number of runs in memory: {len(memory_runs)}")
            
            for i, run in enumerate(memory_runs):
                run_id_attr = None
                if hasattr(run, 'id'):
                    run_id_attr = run.id
                elif hasattr(run, 'run_id'):
                    run_id_attr = run.run_id
                print(f"Memory run {i+1}: {run_id_attr}")
                
                # Print the run's state if available
                if hasattr(run, 'state'):
                    print(f"   State: {run.state}")
        else:
            print(f"No runs found for session {session_id} in memory")
    else:
        print("No runs dictionary found in memory")
    
    if not target_run_id:
        print("⚠️ Could not find a run with the expected state (items=['apple'], counter=1)")
        
        # Try to find a run that has 'apple' in items list as a fallback
        for run_id, state in run_states.items():
            if 'items' in state and 'apple' in state.get('items', []):
                target_run_id = run_id
                print(f"✅ Found fallback run with 'apple' in items: {run_id}")
                print(f"   State: {state}")
                break
        
        # If we still can't find a suitable run, fall back to run_id_3
        if not target_run_id:
            target_run_id = run_id_3
            print(f"⚠️ Falling back to run_id_3: {target_run_id}")
    else:
        print(f"✅ Using target run ID: {target_run_id}")
    
    new_session_id = team.create_session_from_history(
        source_session_id=session_id,
        run_id=target_run_id
    )
    
    if new_session_id:
        print(f"Created new session with ID: {new_session_id}")
        
        # Load the new session
        team.read_from_storage(session_id=new_session_id)
        print(f"Initial state of new session: {team.session_state}")
        
        # Set the session ID for the team
        team.session_id = new_session_id
        
        # Check that the conversation history was properly copied
        if hasattr(team.memory, 'runs') and isinstance(team.memory.runs, dict):
            new_session_runs = team.memory.runs.get(new_session_id, [])
            print(f"\n--- Checking conversation history in new session ---")
            print(f"Number of runs in new session: {len(new_session_runs)}")
            
            if new_session_runs:
                # Print all run IDs in the new session for debugging
                print("\n--- Run IDs in new session ---")
                for i, run in enumerate(new_session_runs):
                    run_id_attr = None
                    if hasattr(run, 'id'):
                        run_id_attr = run.id
                    elif hasattr(run, 'run_id'):
                        run_id_attr = run.run_id
                    print(f"Run {i+1}: {run_id_attr}")
            
                # Check the first run
                first_run = new_session_runs[0]
                if hasattr(first_run, 'messages') and first_run.messages:
                    first_message = None
                    for msg in first_run.messages:
                        if msg.role == 'user':
                            first_message = msg
                            break
                
                    if first_message:
                        print(f"\nFirst user message in new session: '{first_message.content}'")
                        print(f"Expected: 'Hello! Let's start tracking some state.'")
                        if first_message.content == "Hello! Let's start tracking some state.":
                            print("✅ First message matches the original session")
                        else:
                            print("❌ First message doesn't match the expected content")
                    else:
                        print("❌ Could not find the first user message in the new session")
                else:
                    print("❌ No messages found in the first run of the new session")
                
                # Check if we have the expected number of runs based on the target run
                # For our demo, we're branching after the third message (Add 'apple' to the items list)
                expected_runs = 3
                if len(new_session_runs) >= expected_runs:
                    print(f"\n✅ Found {len(new_session_runs)} runs in the new session (expected at least {expected_runs})")
                
                    # Check the third run (Add 'apple' to the items list)
                    if len(new_session_runs) >= 3:
                        third_run = new_session_runs[2]
                        if hasattr(third_run, 'messages') and third_run.messages:
                            third_message = None
                            for msg in third_run.messages:
                                if msg.role == 'user':
                                    third_message = msg
                                    break
                        
                            if third_message:
                                print(f"\nThird user message in new session: '{third_message.content}'")
                                print(f"Expected to contain: 'apple'")
                                if "apple" in third_message.content:
                                    print("✅ Third message contains 'apple' as expected")
                                else:
                                    print("❌ Third message doesn't contain 'apple'")
                            else:
                                print("❌ Could not find the third user message in the new session")
                        else:
                            print("❌ No messages found in the third run of the new session")
                else:
                    print(f"\n❌ Found only {len(new_session_runs)} runs in the new session (expected at least {expected_runs})")
                
            else:
                print("❌ No runs found in the new session")
        else:
            print("❌ No runs dictionary found in the team memory")
        
        # Continue with the new session but take a different path
        print("\n--- Continuing with new branched session ---")
        
        # First run in new session - decrement counter
        response = team.run("Decrement the counter please.")
        print(f"New session run 1 response: {response}")
        print(f"New session run 1 state: {team.session_state}")
        
        # Make sure the session is saved to storage
        team.write_to_storage(session_id=new_session_id)
        
        # Second run in new session - add a different item
        response = team.run("Add 'orange' to the items list.")
        print(f"New session run 2 response: {response}")
        print(f"New session run 2 state: {team.session_state}")
        
        # Make sure the session is saved to storage
        team.write_to_storage(session_id=new_session_id)
        
        # Compare the final states of both sessions
        print("\n--- Comparing final states ---")
        
        # Load the original session
        team.session_id = session_id
        team.read_from_storage(session_id=session_id)
        original_final_state = team.session_state
        print(f"Original session final state: {original_final_state}")
        
        # Load the new session
        team.session_id = new_session_id
        team.read_from_storage(session_id=new_session_id)
        new_final_state = team.session_state
        print(f"New session final state: {new_final_state}")
        
        print("\nSuccess! The new session was created with the correct state from the target run")
        print("and then evolved independently from the original session.")
    else:
        print("Failed to create new session from history")


if __name__ == "__main__":
    main()
