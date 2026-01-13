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
import os

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


# Helper: parses MCP result payload into a normalized dict
def parse_mcp_result(data: dict) -> dict:
    result = data.get("result", {})
    if "content" in result:
        parsed = json.loads(result["content"][0]["text"])
        return parsed
    return result


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


from collections import deque
from datetime import datetime, timezone

# Shared memory: stores recent GCP actions for reasoning agent
ACTION_MEMORY: deque[dict] = deque(maxlen=500)


# Records a GCP tool action into shared in-memory store
def _record_gcp_action(kind: str, project: str, zone: str, name: str | None, payload: dict, result: dict) -> None:
    ACTION_MEMORY.append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "project": project,
            "zone": zone,
            "name": name,
            "payload": payload,
            "result": result,
        }
    )


# Tool: returns the most recent GCP actions to reasoning agent
@function_tool
async def get_recent_gcp_actions(limit: int = 20) -> dict:
    return {"actions": list(ACTION_MEMORY)[-limit:]}


# Starts the internal MCP server for GCP operations
def start_compute_mcp_server():
    uvicorn.run(
        compute_mcp_app,
        host="127.0.0.1",
        port=3334,
        log_level="warning",
    )


# Cleans up external clients when shutting down
async def _cleanup_clients() -> None:
    await async_weaviate_client.close()
    await async_openai_client.close()


# Handles SIGINT graceful shutdown and cleanup
def _handle_sigint(signum: int, frame: object) -> None:
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


# Main request handler for each user message
async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    reasoning_agent = agents.Agent(
        name="Reliabot",
        instructions=REACT_INSTRUCTIONS,
        tools=[
            agents.function_tool(async_knowledgebase.search_knowledgebase),
            get_recent_gcp_actions,
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client,
        ),
    )

    # Loads project context (project ID + zone) from .env-style file
    @function_tool
    async def load_project_context(alias: str) -> dict:
        path = Path(__file__).parent / "contexts" / f"{alias.lower()}.env"
        if not path.exists():
            raise RuntimeError(f"Unknown project context: {alias}")

        data = {}
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()

        return {"project": data["PROJECT_ID"], "zone": data["ZONE"]}

    # Lists GCP instances through MCP server
    @function_tool
    async def gcp_list_instances(project: str, zone: str) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": "gcp-list",
            "method": "tools/call",
            "params": {"name": "list_instances", "arguments": {"project": project, "zone": zone}},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("http://127.0.0.1:3334/mcp", content=json.dumps(payload), headers={"Content-Type": "application/json"})

        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])

        parsed = parse_mcp_result(data)

        _record_gcp_action("list_instances", project, zone, None, {"project": project, "zone": zone}, parsed)
        return parsed

    # Starts a GCP VM through MCP server
    @function_tool
    async def gcp_start_instance(project: str, zone: str, name: str) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": "gcp-start",
            "method": "tools/call",
            "params": {"name": "start_instance", "arguments": {"project": project, "zone": zone, "name": name}},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("http://127.0.0.1:3334/mcp", content=json.dumps(payload), headers={"Content-Type": "application/json"})

        r.raise_for_status()
        parsed = parse_mcp_result(r.json())

        _record_gcp_action("start_instance", project, zone, name, {"project": project, "zone": zone, "name": name}, parsed)
        return parsed

    # Stops a GCP VM through MCP server
    @function_tool
    async def gcp_stop_instance(project: str, zone: str, name: str) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": "gcp-stop",
            "method": "tools/call",
            "params": {"name": "stop_instance", "arguments": {"project": project, "zone": zone, "name": name}},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("http://127.0.0.1:3334/mcp", content=json.dumps(payload), headers={"Content-Type": "application/json"})

        r.raise_for_status()
        parsed = parse_mcp_result(r.json())

        _record_gcp_action("stop_instance", project, zone, name, {"project": project, "zone": zone, "name": name}, parsed)
        return parsed

    # Main orchestrator agent that coordinates context loading + VM actions
    main_agent = agents.Agent(
        name="MainAgent",
        instructions="""
You are the main agent.

If a project alias is mentioned:
1. Load its context
2. List instances
3. If the user asks to start a VM:
   - Choose a TERMINATED instance
   - Start it
4. If the user asks to stop a VM:
   - Choose a RUNNING instance
   - Stop it

You have access to recent GCP actions.
""",
        tools=[
            reasoning_agent.as_tool(tool_name="reasoning_agent", tool_description="Search knowledge base + GCP action history."),
            load_project_context,
            gcp_list_instances,
            gcp_start_instance,
            gcp_stop_instance,
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client,
        ),
    )

    with langfuse_client.start_as_current_span(name="Reliabot") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)

        async for item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(item)
            if gr_messages:
                yield gr_messages

        span.update(output=result_stream.final_output)

    pretty_print(gr_messages)
    yield gr_messages


# Creates and launches a Gradio ChatInterface for the application
demo = gr.ChatInterface(
    _main,
    title="Reliabot Alpha v1",
    type="messages",
    theme="glass",
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
