#!/usr/bin/env python3
"""
Local MCP server for GCP Compute Engine (API-backed, lazy init).

IMPORTANT:
- This MCP client ONLY supports TextContent in CallToolResult
- Structured JSON MUST be serialized into text
"""

import json
import sys
from typing import Any, Dict

from googleapiclient.discovery import build
from google.auth import default


# ─────────────────────────────────────────────
# MCP helpers
# ─────────────────────────────────────────────

def send(message: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def send_error(req_id: int, code: int, message: str) -> None:
    send({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": code,
            "message": message,
        },
    })


# ─────────────────────────────────────────────
# Lazy GCP client
# ─────────────────────────────────────────────

_compute = None

def get_compute():
    global _compute
    if _compute is None:
        credentials, _ = default()
        _compute = build(
            "compute",
            "v1",
            credentials=credentials,
            cache_discovery=False,
        )
    return _compute


# ─────────────────────────────────────────────
# MCP handlers
# ─────────────────────────────────────────────

def handle_initialize(req_id: int) -> None:
    send({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {
                "name": "gcp-compute-mcp-api",
                "version": "0.6.0",
            },
        },
    })


def handle_tools_list(req_id: int) -> None:
    send({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "tools": [
                {
                    "name": "list_instances",
                    "description": "List GCP Compute Engine instances",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "project": {"type": "string"},
                            "zone": {"type": "string"},
                        },
                        "required": ["project", "zone"],
                    },
                }
            ]
        },
    })


def handle_tools_call(req_id: int, params: Dict[str, Any]) -> None:
    tool_name = params.get("name")
    args = params.get("arguments", {})

    if tool_name != "list_instances":
        send_error(req_id, -32601, f"Unknown tool: {tool_name}")
        return

    project = args.get("project")
    zone = args.get("zone")

    if not project or not zone:
        send_error(req_id, -32602, "Both 'project' and 'zone' are required")
        return

    try:
        compute = get_compute()
        response = compute.instances().list(
            project=project,
            zone=zone,
            maxResults=50,
        ).execute()

        instances = response.get("items", [])

        # IMPORTANT: TextContent ONLY
        payload = json.dumps(
            {"instances": instances},
            indent=2,
        )

        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": payload,
                    }
                ]
            },
        })

    except Exception as exc:
        send_error(req_id, -32000, str(exc))


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = request.get("id")
        method = request.get("method")

        if method == "initialize":
            handle_initialize(req_id)

        elif method == "tools/list":
            handle_tools_list(req_id)

        elif method == "tools/call":
            handle_tools_call(req_id, request.get("params", {}))

        else:
            send_error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
