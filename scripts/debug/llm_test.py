import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "mistral",
    "prompt": "Say hi",
    "max_tokens": 10
}
res = requests.post(url, json=payload)
print(res.status_code)
print(res.text[:100])
