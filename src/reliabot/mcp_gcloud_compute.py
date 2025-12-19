import json, subprocess, sys

for line in sys.stdin:
    req = json.loads(line)

    if req["type"] == "list_tools":
        print(json.dumps({
            "tools": {
                "list_instances": {}
            }
        }), flush=True)

    if req["type"] == "call_tool":
        out = subprocess.check_output([
            "gcloud", "compute", "instances", "list",
            "--format=json"
        ])
        print(json.dumps({"result": json.loads(out)}), flush=True)
