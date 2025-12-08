import requests
import json

token = "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"
repo_name = "adhimiw"
url = "https://api.github.com/user/repos"

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

data = {
    "name": repo_name,
    "private": True
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 201:
    print(f"Successfully created repository: {repo_name}")
    print(response.json()['clone_url'])
elif response.status_code == 422:
    print(f"Repository already exists or validation failed: {response.json().get('message')}")
else:
    print(f"Failed to create repository. Status code: {response.status_code}")
    print(response.text)
