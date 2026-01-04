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


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_compute_client():
    global _compute_client
    if _compute_client is None:
        _compute_client = build("compute", "v1")
    return _compute_client


def jsonrpc_result(result: Any, id: str):
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    }


def jsonrpc_error(message: str, id: str):
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": -32000,
            "message": message,
        },
    }


# ─────────────────────────────────────────────
# Compute handlers
# ─────────────────────────────────────────────

def list_instances(project: str, zone: str) -> Dict[str, Any]:
    compute = get_compute_client()
    resp = compute.instances().list(
        project=project,
        zone=zone,
    ).execute()

    instances = []
    for inst in resp.get("items", []):
        instances.append(
            {
                "name": inst["name"],
                "status": inst["status"],
            }
        )

    return {
        "instances": instances,
    }


def start_instance(project: str, zone: str, name: str) -> Dict[str, Any]:
    compute = get_compute_client()

    op = (
        compute.instances()
        .start(
            project=project,
            zone=zone,
            instance=name,
        )
        .execute()
    )

    return {
        "status": "STARTING",
        "message": f"Start request sent for VM '{name}'",
        "operation": op.get("name"),
    }


# ✅ MINIMAL ADD: fire-and-forget stop
def stop_instance(project: str, zone: str, name: str) -> Dict[str, Any]:
    compute = get_compute_client()

    op = (
        compute.instances()
        .stop(
            project=project,
            zone=zone,
            instance=name,
        )
        .execute()
    )

    return {
        "status": "STOPPING",
        "message": f"Stop request sent for VM '{name}'",
        "operation": op.get("name"),
    }


# ─────────────────────────────────────────────
# MCP endpoint
# ─────────────────────────────────────────────

@app.post("/mcp")
async def mcp(request: Request):
    payload = await request.json()

    rpc_id = payload.get("id")
    params = payload.get("params", {})
    tool_name = params.get("name")
    args = params.get("arguments", {})

    try:
        if tool_name == "list_instances":
            result = list_instances(
                project=args["project"],
                zone=args["zone"],
            )

        elif tool_name == "start_instance":
            result = start_instance(
                project=args["project"],
                zone=args["zone"],
                name=args["name"],
            )

        # ✅ MINIMAL ADD: tool routing
        elif tool_name == "stop_instance":
            result = stop_instance(
                project=args["project"],
                zone=args["zone"],
                name=args["name"],
            )

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
