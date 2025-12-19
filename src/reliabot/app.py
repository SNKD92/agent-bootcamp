"""Reliabot.

Log traces to LangFuse for observability and evaluation.
"""

import asyncio
import contextlib
import signal
import sys

import agents
import gradio as gr
import os
from agents.mcp import MCPServerStdio
from agents.mcp import create_static_tool_filter
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.prompts import REACT_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)

set_up_logging()

AGENT_LLM_NAME = "gemini-2.5-flash"

configs = Configs.from_env_var()
async_weaviate_client = get_weaviate_async_client(
    http_host=configs.weaviate_http_host,
    http_port=configs.weaviate_http_port,
    http_secure=configs.weaviate_http_secure,
    grpc_host=configs.weaviate_grpc_host,
    grpc_port=configs.weaviate_grpc_port,
    grpc_secure=configs.weaviate_grpc_secure,
    api_key=configs.weaviate_api_key,
)
async_openai_client = AsyncOpenAI()
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="Devops_reasoning_traces",
)


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    repo_path = os.path.abspath("/home/coder/agent-bootcamp/")

    reasoning_agent = agents.Agent(
        name="Reliabot",
        instructions=REACT_INSTRUCTIONS,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME, openai_client=async_openai_client
        ),
    )
    async with MCPServerStdio(
        name="Git server",
        params={
            "command": "uvx",
            "args": ["mcp-server-git"],
        },
     )as mcp_server:
        git_agent = agents.Agent(
            name="Git Assistant",
            instructions=f"Answer questions about the git repository at {repo_path}, use that for repo_path",
            mcp_servers=[mcp_server],
            model=agents.OpenAIChatCompletionsModel(
                model=AGENT_LLM_NAME, openai_client=async_openai_client
            ))
    async with MCPServerStdio(
        name="GCP Compute server",
        params={
            "command": "python",
            "args": ["/home/coder/agent-bootcamp/src/reliabot/mcp_gcloud_compute.py"],
        },
    ) as compute_mcp_server:
        compute_agent = agents.Agent(
            name="GCP Compute Assistant",
            instructions="Answer questions about GCP Compute Engine using the MCP server.",
            mcp_servers=[compute_mcp_server],
            model=agents.OpenAIChatCompletionsModel(
                model=AGENT_LLM_NAME, openai_client=async_openai_client
            ),
        )
        main_agent = agents.Agent(
            name="MainAgent",
            instructions="""
                You are the main agent. You orchestrating git agent and reasoning agent. When user asks about files check the git repo.
                When user asks about issue check the reasoning agent.
            """,
            # Allow the planner agent to invoke the worker agent.
            # The long context provided to the worker agent is hidden from the main agent.
            tools=[
                reasoning_agent.as_tool(
                    tool_name="reasoning_agent",
                    tool_description=(
                        "Search the knowledge base for a query and return a concise summary "
                        "of the key findings, along with the sources used to generate "
                        "the summary"
                    ),
                ),
                git_agent.as_tool(
                    tool_name="Git_MCP_server",
                    tool_description=(
                        "Search the Git repo for a info about the issue "
                    ),  
                ),
                compute_agent.as_tool(
                    tool_name="GCP_Compute_MCP",
                    tool_description="Query GCP Compute Engine (instances, zones, metadata) via MCP",
                ),
            ],
            # a larger, more capable model for planning and reasoning over summaries
            model=agents.OpenAIChatCompletionsModel(
                model=AGENT_LLM_NAME, openai_client=async_openai_client
            ),
        )
        with langfuse_client.start_as_current_span(name="Reliabot") as span:
            span.update(input=question)

            result_stream = agents.Runner.run_streamed(main_agent, input=question)
            async for _item in result_stream.stream_events():
                gr_messages += oai_agent_stream_to_gradio_messages(_item)
                if len(gr_messages) > 0:
                    yield gr_messages

            span.update(output=result_stream.final_output)

        pretty_print(gr_messages)
        yield gr_messages



demo = gr.ChatInterface(
    _main,
    title="0.1beta Reliabot + LangFuse",
    type="messages",
    examples=[
        "Where in CI/CD pipeline the job stuck when GitLab runner lose connection during deploy?"
        "Which microservice caused latency spike when Istio sidecar injected wrongly configured certificates?",
    ],
)


if __name__ == "__main__":
    configs = Configs.from_env_var()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
