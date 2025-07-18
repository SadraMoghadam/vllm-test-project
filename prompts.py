import requests

data = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
}

res = requests.post("http://localhost:8000/generate", json=data)
print(res.json())
