# Керівництво репозиторію

## Структура проєкту та модулі
- `src/`: основна логіка — `smc_detector.py` (SMC‑примітиви), `signal_generator.py` (збирання сигналів), `live_smc_engine.py`/`live_monitor.py` (рантайм), `data_loader.py` (CSV I/O), `models.py` (dataclasses).
- `main.py`: точка входу бек‑тест/CSV. `live_trading.py`: точка входу монітора в реальному часі.
- `config.py`: центральні налаштування/дефолти. `requirements.txt`: залежності. `signals.csv`/`live_trading.log`: результати/логи.

## Збирання, тести, запуск
- Створити середовище: `python -m venv venv`
- Встановити залежності (Windows): `venv\Scripts\python.exe -m pip install -r requirements.txt`
- Встановити залежності (Unix): `source venv/bin/activate && pip install -r requirements.txt`
- Запуск бек‑тесту: `python main.py --ltf data/btc_15m.csv --htf data/btc_4h.csv`
- Запуск live‑монітора: `python live_trading.py --symbol ETHUSDT --desktop-alerts`
- Тести (pytest): `pytest -q` (за потреби: `pip install pytest`)

## Стиль коду та іменування
- Python, PEP 8, відступ 4 пробіли. Використовуйте type hints за можливості.
- Імена: модулі/функції/змінні — `snake_case`; класи — `PascalCase`; константи — `UPPER_SNAKE`.
- Функції мають бути невеликими, «чистими» і тестованими. Надавайте перевагу docstring.

## Рекомендації з тестування
- Фреймворк: pytest. Файли тестів у `tests/` з іменами `test_*.py` (якщо є).
- Для алгоритмів SMC використовуйте синтетичні DataFrame та `monkeypatch` для ізоляції детекторів.
- Запуск швидких перевірок: `pytest -q`; покриття (опційно): `pytest --cov=src -q`.

## Коміти та Pull Request’и
- Повідомлення: наказова форма, лаконічно (напр., `Add tests for FVG detection`).
- PR містить: мету, опис змін, як запускати/валідувати (команди), пов’язані issue. Для TUI — скриншоти/гіфки.
- Тримайте PR вузькими; не змішуйте несуміжні зміни.

## Безпека та конфігурація
- Не комітьте секрети. Передавайте токени через CLI (`--telegram-token`, `--telegram-chat-id`) або змінні середовища.
- `.env.example` — шаблон; створіть локальний `.env` (не комітиться) з `TELEGRAM_TOKEN` і `TELEGRAM_CHAT_ID`.
- Налаштування через `config.py` або CLI; не хардкодьте шляхи під `data/`.

## Огляд архітектури
- Конвеєр: `data_loader` → `smc_detector` → `signal_generator` → (CSV/вивід або `live_monitor` UI/алерти). Тримайте межі чистими і відокремлюйте I/O від логіки.

