#!/usr/bin/env python3
"""
Local MCP HTTP server for GCP Compute Engine (API-backed, lazy init).

IMPORTANT:
- This MCP server ONLY supports TextContent in CallToolResult
- Structured JSON MUST be serialized into text
"""

import json
from typing import Any, Dict

from fastapi import FastAPI, Request
from googleapiclient.discovery import build
from google.auth import default

app = FastAPI()

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
# MCP helpers
# ─────────────────────────────────────────────

def mcp_error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


# ─────────────────────────────────────────────
# MCP HTTP endpoint
# ─────────────────────────────────────────────

@app.post("/mcp")
def handle_mcp(request: Request):
    body = request.json() if hasattr(request, "json") else {}

    try:
        body = json.loads(request._body.decode())
    except Exception:
        return mcp_error(None, -32700, "Invalid JSON")

    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    # ── initialize ───────────────────────────
    if method == "initialize":
        return {
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
        }

    # ── tools/list ───────────────────────────
    if method == "tools/list":
        return {
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
        }

    # ── tools/call ───────────────────────────
    if method == "tools/call":
        tool = params.get("name")
        args = params.get("arguments", {})

        if tool != "list_instances":
            return mcp_error(req_id, -32601, f"Unknown tool: {tool}")

        project = args.get("project")
        zone = args.get("zone")

        if not project or not zone:
            return mcp_error(req_id, -32602, "Both 'project' and 'zone' are required")

        try:
            compute = get_compute()
            response = compute.instances().list(
                project=project,
                zone=zone,
                maxResults=50,
            ).execute()

            instances = response.get("items", [])

            payload = json.dumps(
                {"instances": instances},
                indent=2,
            )

            return {
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
            }

        except Exception as exc:
            return mcp_error(req_id, -32000, str(exc))

    # ── fallback ─────────────────────────────
    return mcp_error(req_id, -32601, f"Method not found: {method}")
