from unittest.mock import patch

from src.telegram import send_telegram_message


def test_send_telegram_message_success():
    token = 'TOKEN'
    chat_id = 'CHAT'
    text = 'hello'
    with patch('src.telegram.requests.post') as mock_post:
        mock_post.return_value.ok = True
        assert send_telegram_message(token, chat_id, text) is True
        mock_post.assert_called_once_with(
            f'https://api.telegram.org/bot{token}/sendMessage',
            data={'chat_id': chat_id, 'text': text},
            timeout=10
        )
