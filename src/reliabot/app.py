"""
Reliabot.

Log traces to LangFuse for observability and evaluation.
"""

import asyncio
import contextlib
import signal
import sys
import threading

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

import uvicorn
import httpx
from agents import function_tool

from mcp_gcloud_compute_http import app as compute_mcp_app

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


# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Start HTTP GCP MCP server
# ─────────────────────────────────────────────

def start_compute_mcp_server():
    uvicorn.run(
        compute_mcp_app,
        host="127.0.0.1",
        port=3334,
        log_level="warning",
    )


# ─────────────────────────────────────────────
# Cleanup / signals
# ─────────────────────────────────────────────

async def _cleanup_clients() -> None:
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


# ─────────────────────────────────────────────
# Main app logic
# ─────────────────────────────────────────────

async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    reasoning_agent = agents.Agent(
        name="Reliabot",
        instructions=REACT_INSTRUCTIONS,
        tools=[agents.function_tool(async_knowledgebase.search_knowledgebase)],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client,
        ),
    )

    # ─────────────────────────────────────────
    # GCP HTTP tool (NO MCP abstraction)
    # ─────────────────────────────────────────

    @function_tool
    async def gcp_list_instances(project: str, zone: str) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "http://127.0.0.1:3334/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": "gcp-list",
                    "method": "tools/call",
                    "params": {
                        "name": "list_instances",
                        "arguments": {
                            "project": project,
                            "zone": zone,
                        },
                    },
                },
            )
            r.raise_for_status()
            return r.json()

    # ─────────────────────────────────────────
    # Main orchestrator agent
    # ─────────────────────────────────────────

    main_agent = agents.Agent(
        name="MainAgent",
        instructions="""
You are the main agent.
- Incidents / history → reasoning_agent
- Live GCP state → gcp_list_instances
""",
        tools=[
            reasoning_agent.as_tool(
                tool_name="reasoning_agent",
                tool_description="Search historical knowledge base.",
            ),
            gcp_list_instances,
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client,
        ),
    )

    # ─────────────────────────────────────────
    # Run + tracing
    # ─────────────────────────────────────────

    with langfuse_client.start_as_current_span(name="Reliabot") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(
            main_agent,
            input=question,
        )

        async for item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(item)
            if gr_messages:
                yield gr_messages

        span.update(output=result_stream.final_output)

    pretty_print(gr_messages)
    yield gr_messages


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

demo = gr.ChatInterface(
    _main,
    title="0.1beta Reliabot + LangFuse",
    type="messages",
    examples=[
        "List compute instances in project project-e7b71f6e-c56d-438f-a7e zone us-central1-a",
    ],
)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)

    # ✅ START MCP HTTP SERVER ONCE (server process only)
    threading.Thread(
        target=start_compute_mcp_server,
        daemon=True,
    ).start()

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
