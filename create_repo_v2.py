import requests
import json

token = "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"

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

# Get user info
user_url = "https://api.github.com/user"
user_response = requests.get(user_url, headers=headers)
if user_response.status_code == 200:
    username = user_response.json()['login']
    print(f"Authenticated as: {username}")
else:
    print("Failed to get user info")
    exit()

repo_name = "reserqach-agent"
url = "https://api.github.com/user/repos"

data = {
    "name": repo_name,
    "private": True
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 201:
    print(f"Successfully created repository: {repo_name}")
    print(response.json()['clone_url'])
elif response.status_code == 422:
    print(f"Repository already exists: {response.json().get('message')}")
else:
    print(f"Failed to create repository. Status code: {response.status_code}")
    print(response.text)
