"""Utility for sending Telegram notifications."""
import logging
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
        if resp.ok:
            return True
        # Try to log Telegram error details if present
        try:
            data = resp.json()
            desc = data.get('description')
        except Exception:
            desc = resp.text[:200]
        logging.warning("Telegram send failed: status=%s, detail=%s", resp.status_code, desc)
        return False
    except requests.RequestException as e:
        logging.error("Telegram request error: %s", e)
        return False

        return resp.ok
    except requests.RequestException:
        return False
