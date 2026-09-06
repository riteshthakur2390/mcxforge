import requests


def build_telegram_request(bot_token: str, chat_id: int, message: str) -> tuple[str, dict]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
    }
    return url, payload


def test_build_telegram_request_shape():
    url, payload = build_telegram_request("token", 1234566, "Test message from your trading bot")

    assert url == "https://api.telegram.org/bottoken/sendMessage"
    assert payload == {
        "chat_id": 1234566,
        "text": "Test message from your trading bot",
    }


def test_requests_post_can_be_mocked(monkeypatch):
    calls = {}

    def fake_post(url, data=None, **kwargs):
        calls["url"] = url
        calls["data"] = data
        calls["kwargs"] = kwargs

        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"ok": True}

        return Response()

    monkeypatch.setattr(requests, "post", fake_post)

    url, payload = build_telegram_request("token", 1234566, "ping")
    response = requests.post(url, data=payload, timeout=5)

    assert response.json()["ok"] is True
    assert calls["url"] == url
    assert calls["data"] == payload
    assert calls["kwargs"]["timeout"] == 5
