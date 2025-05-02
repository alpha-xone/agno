import json
from dataclasses import dataclass
from typing import Iterator, List, Optional, Union, Dict, Any

from pydantic import BaseModel
from agno.utils.log import log_error, log_debug
from agno.agent import Agent
from agno.workflow import Workflow


@dataclass
class SequentialWorkFlow(Workflow):
    """A workflow that runs agents in sequence."""

    agents: Optional[List[Agent]] = None

    def __or__(self, *others):
        return sequential_chain(self, *others)

    def run(self, input_message: str) -> str:
        """
        Execute the sequential chain flow between multiple agents.
        Logs and raises errors if any agent's response is invalid.
        """
        if not self.agents:
            raise ValueError("No agents provided for the sequential workflow.")

        current_message = input_message
        all_responses: Dict[str, Any] = {}  # To store responses from all agents for debugging

        try:
            # Iterate through the chain of agents
            for index, agent in enumerate(self.agents):
                key = agent.name or agent.agent_id or f"Agent_{index}"
                log_debug(f"Running agent '{key}' with input: {current_message}")

                # Run the current agent
                response = agent.run(message=current_message, stream=False)

                # Handle RunResponse or Iterator[RunResponse]
                if isinstance(response, Iterator):
                    # Consume the iterator to get the final response
                    response = list(response)[
                        -1
                    ]  # Get the last response from the iterator

                if not response or not hasattr(response, "content"):
                    raise ValueError(f"Agent '{key}' returned an invalid response.")

                # Serialize the response content
                if isinstance(response.content, BaseModel):
                    # Use model_dump if the content is a response model
                    all_responses[key] = response.content.model_dump()
                elif isinstance(response.content, str):
                    # Use the string content directly
                    all_responses[key] = response.content
                else:
                    # Fallback to JSON serialization for other types
                    try:
                        all_responses[key] = json.dumps(response.content)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to serialize response from agent '{key}': {e}"
                        )

                # Log the response
                log_debug(f"Agent '{key}' response: {all_responses[key]}")

                # Prepare the input for the next agent as a JSON string
                current_message = json.dumps(all_responses)

            # Return the final response
            return current_message

        except Exception as e:
            # Log the error and include the chain of responses for debugging
            log_error(f"Error in sequential chain: {e}")
            log_error(f"Responses so far: {all_responses}")
            raise


def sequential_chain(
    left: Union[Agent, SequentialWorkFlow],
    *others: Union[Agent, SequentialWorkFlow],
) -> Any:
    """
    Create a workflow that runs agents in sequence.
    If the first argument is an agent, it will be wrapped in a workflow.
    """
    # Create a workflow that stores agents
    if isinstance(left, Agent):
        workflow = SequentialWorkFlow(agents=[left])
    else:
        workflow = left

    if not isinstance(workflow.agents, list):
        workflow.agents = []
    for agent in others:
        if isinstance(agent, Agent):
            workflow.agents.append(agent)
        else:
            raise ValueError("OR operator only accepts agents")

    return workflow
