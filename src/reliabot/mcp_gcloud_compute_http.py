"""
Minimal MCP-style HTTP server for Google Cloud Compute Engine.

Exposes basic VM operations over JSON-RPC for local tooling / agents.
"""

import json
from typing import Any, Dict

from fastapi import FastAPI, Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

app = FastAPI()
_compute_client = None


# Returns a cached Google Compute client, creating it on first use
def get_compute_client():
    global _compute_client
    if _compute_client is None:
        _compute_client = build("compute", "v1")
    return _compute_client


# Builds a JSON-RPC success response with result payload
def jsonrpc_result(result: Any, id: str):
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    }


# Builds a JSON-RPC error response with message
def jsonrpc_error(message: str, id: str):
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": -32000,
            "message": message,
        },
    }


# Retrieves list of VM instances with name and status
def list_instances(project: str, zone: str) -> Dict[str, Any]:
    compute = get_compute_client()
    resp = compute.instances().list(project=project, zone=zone).execute()

    instances = [
        {"name": inst["name"], "status": inst["status"]}
        for inst in resp.get("items", [])
    ]

    return {"instances": instances}


# Sends a start request for a specific VM and returns the operation ID
def start_instance(project: str, zone: str, name: str) -> Dict[str, Any]:
    compute = get_compute_client()

    op = compute.instances().start(
        project=project, zone=zone, instance=name
    ).execute()

    return {
        "status": "STARTING",
        "message": f"Start request sent for VM '{name}'",
        "operation": op.get("name"),
    }


# Sends a stop request for a specific VM and returns the operation ID
def stop_instance(project: str, zone: str, name: str) -> Dict[str, Any]:
    compute = get_compute_client()

    op = compute.instances().stop(
        project=project, zone=zone, instance=name
    ).execute()

    return {
        "status": "STOPPING",
        "message": f"Stop request sent for VM '{name}'",
        "operation": op.get("name"),
    }


# Main MCP endpoint that routes JSON-RPC calls to specific compute operations
@app.post("/mcp")
async def mcp(request: Request):
    payload = await request.json()

    rpc_id = payload.get("id")
    params = payload.get("params", {})
    tool_name = params.get("name")
    args = params.get("arguments", {})

    try:
        if tool_name == "list_instances":
            result = list_instances(args["project"], args["zone"])

        elif tool_name == "start_instance":
            result = start_instance(args["project"], args["zone"], args["name"])

        elif tool_name == "stop_instance":
            result = stop_instance(args["project"], args["zone"], args["name"])

        else:
            return jsonrpc_error(f"Unknown tool: {tool_name}", rpc_id)

        return jsonrpc_result(
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result),
                    }
                ]
            },
            rpc_id,
        )

    except HttpError as e:
        return jsonrpc_error(str(e), rpc_id)
    except Exception as e:
        return jsonrpc_error(str(e), rpc_id)
