import requests

def test_upstox():
    UPSTOX_ACCESS_TOKEN='eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1RENDVjYiLCJqdGkiOiI2OWQ4OTUxNWNkMzUwYTU3M2NkMjc5MzQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzc1ODAxNjIyLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzU4NTg0MDB9.N8mFlg_fYqxkneYdgxpc6huhHYj9xEQrJhDKtJK1rU0'

    url = "https://api.upstox.com/v2/user/profile"

    headers = {
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            print("✅ API is working")
            print("User:", response.json().get("data", {}).get("email"))
        else:
            print("❌ API failed")
            print("Status Code:", response.status_code)
            print("Response:", response.text)

    except Exception as e:
        print("❌ Error:", str(e))