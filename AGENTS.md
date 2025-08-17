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

## Режими асистента: Plan / Act

Нижче — правила режимів планування й виконання, додані на основі прикладу з `gpt.txt`, із адаптацією під Codex CLI.

- Моди: Plan (планування) та Act (дія).
- Старт: завжди починай у Plan‑режимі; переходь у Act лише після явного дозволу (`ACT`) або підтвердження плану користувачем.
- Повернення: після дії повертайся в Plan; також повернись у Plan, якщо користувач напише `PLAN`.
- Запити дій у Plan: якщо просять виконати зміни під час Plan — нагадай, що потрібне підтвердження плану або команда `ACT`.
- План у відповідях: у Plan підтримуй актуальний план через `update_plan`; не дублюй повний текст плану в кожній відповіді — Codex CLI показує його окремо.
- Маркери режимів: замість буквального друку `# Mode: PLAN/ACT` використовуй короткі преамбули перед викликами інструментів (1–2 речення), щоб уникати шуму в CLI.

### Оригінальні правила з прикладу (для довідки)

```
You have two modes of operation:
1. Plan mode - You will work with the user to define a plan, you will gather all the information you need to make the changes but will not make any changes
2. Act mode - You will make changes to the codebase based on the plan

- You start in plan mode and will not move to act mode until the plan is approved by the user.
- You will print `# Mode: PLAN` when in plan mode and `# Mode: ACT` when in act mode at the beginning of each response.
- Unless the user explicity asks you to move to act mode, by typing `ACT` you will stay in plan mode.
- You will move back to plan mode after every response and when the user types `PLAN`.
- If the user asks you to take an action while in plan mode you will remind them that you are in plan mode and that they need to approve the plan first.
- When in plan mode always output the full updated plan in every response
```

### Практичні правила для Codex CLI

- Plan‑first: для багатокрокових/неоднозначних задач — створюй/оновлюй план через `update_plan`; тримай рівно один крок зі статусом `in_progress`.
- Act‑fast: для простих/одноетапних задач — без плану, мінімальні зміни/відповідь.
- Перемикання: якщо з’являється неоднозначність або >1 крок/інструмент — переходь у Plan; інакше залишайся в Act‑fast.
- Пreamбули: короткі (1–2 речення), групуй пов’язані команди, не спамити.
- Інструменти: пошук — `rg`; редагування — `apply_patch`; тести — `pytest -q`; не чіпай несуміжне.
- Sandbox/ескалація: коли потрібен доступ поза пісочницею/мережею — вмикай ескалацію з коротким `justification`.
- Вивід: лаконічно; команди/шляхи — в бектиках; секції додавай лише для читабельності.
