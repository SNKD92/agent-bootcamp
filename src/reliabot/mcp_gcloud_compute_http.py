#!/usr/bin/env python3
"""
Local MCP HTTP server for GCP Compute Engine (API-backed, lazy init).

IMPORTANT:
- This MCP server ONLY supports TextContent in CallToolResult
- Structured JSON MUST be serialized into text
"""

import json
import time
import os
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
        print(f"[DEBUG][MCP][PID {os.getpid()}] Initializing GCP Compute client")
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
    print(f"[DEBUG][MCP][PID {os.getpid()}] ERROR {code}: {message}")
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
async def handle_mcp(request: Request):
    try:
        raw = await request.body()
        print(f"[DEBUG][MCP][PID {os.getpid()}] RAW request = {raw.decode()}")
        body = json.loads(raw)
    except Exception as e:
        print(f"[DEBUG][MCP] JSON parse failure: {e}")
        return mcp_error(None, -32700, "Invalid JSON")

    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    print(
        f"[DEBUG][MCP][PID {os.getpid()}] "
        f"method={method} id={req_id} params={params}"
    )

    # ── initialize ───────────────────────────
    if method == "initialize":
        print("[DEBUG][MCP] initialize called")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {
                    "name": "gcp-compute-mcp-api",
                    "version": "0.8.2",
                },
            },
        }

    # ── tools/list ───────────────────────────
    if method == "tools/list":
        print("[DEBUG][MCP] tools/list called")
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
                    },
                    {
                        "name": "start_instance",
                        "description": "Start a GCP Compute Engine instance",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "project": {"type": "string"},
                                "zone": {"type": "string"},
                                "name": {"type": "string"},
                            },
                            "required": ["project", "zone", "name"],
                        },
                    },
                ]
            },
        }

    # ── tools/call ───────────────────────────
    if method == "tools/call":
        tool = params.get("name")
        args = params.get("arguments", {})

        print(f"[DEBUG][MCP] tools/call tool={tool} args={args}")

        compute = get_compute()

        # ── list_instances ───────────────────
        if tool == "list_instances":
            project = args.get("project")
            zone = args.get("zone")

            if not project or not zone:
                return mcp_error(req_id, -32602, "Both 'project' and 'zone' are required")

            print(f"[DEBUG][MCP] Listing instances project={project} zone={zone}")

            response = compute.instances().list(
                project=project,
                zone=zone,
                maxResults=50,
            ).execute()

            payload = json.dumps(
                {"instances": response.get("items", [])},
                indent=2,
            )

            print(f"[DEBUG][MCP] list_instances payload = {payload}")

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": payload}]},
            }

        # ── start_instance ───────────────────
        if tool == "start_instance":
            project = args.get("project")
            zone = args.get("zone")
            name = args.get("name")

            if not project or not zone or not name:
                return mcp_error(req_id, -32602, "project, zone and name are required")

            print(
                f"[DEBUG][MCP] Starting instance "
                f"project={project} zone={zone} name={name}"
            )

            # submit start request
            op = compute.instances().start(
                project=project,
                zone=zone,
                instance=name,
            ).execute()

            print(f"[DEBUG][MCP] start operation submitted: {op}")

            # wait for operation acceptance
            while True:
                op_result = compute.zoneOperations().get(
                    project=project,
                    zone=zone,
                    operation=op["name"],
                ).execute()

                print(f"[DEBUG][MCP] op status = {op_result.get('status')}")

                if op_result["status"] == "DONE":
                    if "error" in op_result:
                        return mcp_error(req_id, -32000, str(op_result["error"]))
                    break

                time.sleep(2)

            # poll instance state (best-effort)
            status = None
            for i in range(30):
                instance = compute.instances().get(
                    project=project,
                    zone=zone,
                    instance=name,
                ).execute()

                status = instance.get("status")
                print(f"[DEBUG][MCP] poll {i} instance status = {status}")

                if status == "RUNNING":
                    break

                time.sleep(5)

            payload = json.dumps(
                {
                    "instance": name,
                    "status": status,
                    "note": "start request accepted; instance may still be provisioning",
                },
                indent=2,
            )

            print(f"[DEBUG][MCP] start_instance payload = {payload}")

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": payload}]},
            }

        return mcp_error(req_id, -32601, f"Unknown tool: {tool}")

    return mcp_error(req_id, -32601, f"Method not found: {method}")
