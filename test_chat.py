import requests
import json

url = "http://localhost:8000/chat"
payload = {
    "query": "Hi! Tell me about artificial intelligence",
    "max_results": 3
}

headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
