
import requests
import json

def test_mcp_post():
    url = "http://127.0.0.1:12306/mcp"
    print(f"Testing POST to {url}...")
    try:
        # Try a basic JSON-RPC initialize
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            }
        }
        response = requests.post(url, json=payload, timeout=5)
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        print(f"Response headers: {response.headers}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_mcp_post()
