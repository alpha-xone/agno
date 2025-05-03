import asyncio

from agno.utils.builder import agno

# Reads from lib/tools.py from repo root
tools = agno("lib/tools")
# Reads from current working directory
model = agno("model")
roles = agno("discussion_team", model=model)

# Reads from lib/teams.py from repo root
agent_team = agno(
    "lib/team",
    name="Discussion Team",
    mode="collaborate",
    model=model,
    members=[
        roles["reddit_researcher"],
        roles["hackernews_researcher"],
        roles["academic_paper_researcher"],
        roles["twitter_researcher"],
    ],
    instructions=[
        "You are a discussion master.",
        "You have to stop the discussion when you think the team has reached a consensus.",
    ],
    success_criteria="The team has reached a consensus.",
    send_team_context_to_members=True,
    update_team_context=True,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    show_members_responses=True,
)

if __name__ == "__main__":
    asyncio.run(
        agent_team.print_response(
            message="Start the discussion on the topic: 'What is the best way to learn to code?'",
            stream=True,
            stream_intermediate_steps=True,
        )
    )
