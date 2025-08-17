"""Utility for sending Telegram notifications."""
import requests


def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
    """Send a Telegram message.

    Args:
        token: Bot token obtained from @BotFather.
        chat_id: ID of the chat to send the message to.
        text: Message text.

    Returns:
        True if the request succeeded, False otherwise.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        return resp.ok
    except requests.RequestException:
        return False
