"""
Reliabot.

Log traces to LangFuse for observability and evaluation.
"""

import asyncio
import contextlib
import signal
import sys
import threading
import json
from pathlib import Path
import multiprocessing
import os  # DEBUG

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
# Helpers
# ─────────────────────────────────────────────

def parse_mcp_result(data: dict) -> dict:
    print(f"[DEBUG][APP][PID {os.getpid()}] parse_mcp_result raw = {data}")
    result = data.get("result", {})
    if "content" in result:
        parsed = json.loads(result["content"][0]["text"])
        print(f"[DEBUG][APP][PID {os.getpid()}] parse_mcp_result parsed = {parsed}")
        return parsed
    return result


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
    print(f"[DEBUG][MCP] Starting MCP server PID={os.getpid()}")
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
    print(f"[DEBUG][APP][PID {os.getpid()}] User question = {question}")
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
    # Context loader tool
    # ─────────────────────────────────────────

    @function_tool
    async def load_project_context(alias: str) -> dict:
        print(f"[DEBUG][APP] load_project_context alias={alias}")
        path = Path(__file__).parent / "contexts" / f"{alias.lower()}.env"
        if not path.exists():
            raise RuntimeError(f"Unknown project context: {alias}")

        data = {}
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()

        ctx = {
            "project": data["PROJECT_ID"],
            "zone": data["ZONE"],
        }
        print(f"[DEBUG][APP] loaded context = {ctx}")
        return ctx

    # ─────────────────────────────────────────
    # GCP MCP tools
    # ─────────────────────────────────────────

    @function_tool
    async def gcp_list_instances(project: str, zone: str) -> dict:
        payload = {
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
        }

        print(f"[DEBUG][APP][PID {os.getpid()}] LIST payload = {payload}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "http://127.0.0.1:3334/mcp",
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )

        print(f"[DEBUG][APP] LIST HTTP status = {r.status_code}")
        print(f"[DEBUG][APP] LIST raw response = {r.text}")

        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return parse_mcp_result(data)

    @function_tool
    async def gcp_start_instance(project: str, zone: str, name: str) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": "gcp-start",
            "method": "tools/call",
            "params": {
                "name": "start_instance",
                "arguments": {
                    "project": project,
                    "zone": zone,
                    "name": name,
                },
            },
        }

        print(f"[DEBUG][APP][PID {os.getpid()}] START payload = {payload}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "http://127.0.0.1:3334/mcp",
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )

        print(f"[DEBUG][APP] START HTTP status = {r.status_code}")
        print(f"[DEBUG][APP] START raw response = {r.text}")

        r.raise_for_status()
        return parse_mcp_result(r.json())

    # ─────────────────────────────────────────
    # Main orchestrator agent
    # ─────────────────────────────────────────

    main_agent = agents.Agent(
        name="MainAgent",
        instructions="""
You are the main agent.

If a project alias is mentioned:
1. Load its context
2. List instances
3. If user asks to start a VM:
   - Choose a TERMINATED instance
   - Start it automatically
""",
        tools=[
            reasoning_agent.as_tool(
                tool_name="reasoning_agent",
                tool_description="Search historical knowledge base.",
            ),
            load_project_context,
            gcp_list_instances,
            gcp_start_instance,
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client,
        ),
    )

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
        "Start a VM in Denis",
        "Bring up the VM in Amandeep",
        "Start the stopped instance in Rameesha",
    ],
)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)

    multiprocessing.Process(
        target=start_compute_mcp_server,
        daemon=True,
    ).start()

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
