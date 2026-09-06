"""
scripts/detect_channel_id.py — Auto-Detect Telegram Channel ID
=============================================================
Listens for posts or administrator addition events on @mcxforge_bot
(8971620178:AAHhLt1kw-CiF6cLgYbPj5F0aDzFMz4FYyk) and automatically
saves the channel chat_id to mcxforge/.env.

Usage:
    python scripts/detect_channel_id.py
"""

import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("LIVE_TELEGRAM_BOT_TOKEN", "8971620178:AAHhLt1kw-CiF6cLgYbPj5F0aDzFMz4FYyk")


def main():
    print(f"📡 Polling @mcxforge_bot for channel events...")
    print(f"👉 Please post any message in your Telegram Channel where @mcxforge_bot is an Admin.\n")

    last_update_id = 0
    start_time = time.time()

    while True:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            params = {
                "offset": last_update_id + 1,
                "timeout": 10,
                "allowed_updates": ["message", "channel_post", "my_chat_member", "chat_member"],
            }
            res = requests.get(url, params=params, timeout=15).json()

            if not res.get("ok"):
                print(f"Telegram API error: {res}")
                time.sleep(2)
                continue

            updates = res.get("result", [])
            for u in updates:
                last_update_id = u["update_id"]
                chat = None
                source = "unknown"

                if "channel_post" in u:
                    chat = u["channel_post"].get("chat")
                    source = "channel_post"
                elif "my_chat_member" in u:
                    chat = u["my_chat_member"].get("chat")
                    source = "my_chat_member"
                elif "message" in u:
                    chat = u["message"].get("chat")
                    source = "message"

                if chat:
                    chat_id = str(chat.get("id"))
                    title = chat.get("title", chat.get("first_name", "Unknown"))
                    chat_type = chat.get("type", "unknown")
                    print(f"\n🎉 Detected Chat Event ({source})!")
                    print(f"  • Title: {title}")
                    print(f"  • Type:  {chat_type}")
                    print(f"  • ID:    {chat_id}")

                    if chat_type in ("channel", "supergroup"):
                        print(f"\n✅ Saving LIVE_TELEGRAM_CHAT_ID={chat_id} to .env...")
                        _update_env("LIVE_TELEGRAM_CHAT_ID", chat_id)

                        # Send test message to verified channel
                        test_msg = (
                            f"🔔 *MCXForge Alert Channel Connected*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Channel: {title}\n"
                            f"Chat ID: `{chat_id}`\n"
                            f"Status: Live & Connected ✅"
                        )
                        requests.post(
                            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                            json={"chat_id": chat_id, "text": test_msg, "parse_mode": "Markdown"},
                            timeout=5,
                        )
                        print(f"🚀 Test message dispatched successfully to {title} ({chat_id})!")
                        return

        except Exception as exc:
            print(f"Polling warning: {exc}")

        if time.time() - start_time > 60:
            print(".", end="", flush=True)
            start_time = time.time()
        time.sleep(1)


def _update_env(key: str, value: str):
    env_path = BASE_DIR / ".env"
    lines = []
    found = False
    with open(env_path, "r") as f:
        for line in f:
            if line.startswith(f"{key}="):
                lines.append(f"{key}={value}\n")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
