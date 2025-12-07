import requests
import sys

try:
    print("Testing /sse endpoint...")
    with requests.get("http://127.0.0.1:12306/sse", stream=True, timeout=5) as r:
        print(f"Status: {r.status_code}")
        print(f"Headers: {r.headers}")
        for line in r.iter_lines():
            if line:
                print(f"Received: {line}")
                break # Just need to see one line to confirm it's working
except Exception as e:
    print(f"Error connecting to /sse: {e}")

try:
    print("\nTesting /mcp endpoint...")
    with requests.get("http://127.0.0.1:12306/mcp", stream=True, timeout=5) as r:
        print(f"Status: {r.status_code}")
        print(f"Headers: {r.headers}")
        print(f"Content: {r.text[:100]}")
except Exception as e:
    print(f"Error connecting to /mcp: {e}")
