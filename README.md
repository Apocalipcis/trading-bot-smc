# SMC Trading Bot - Smart Money Concepts Trading System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

Розумна торгова система на основі Smart Money Concepts (SMC) для аналізу криптовалютних ринків з веб-інтерфейсом, автоматичним бектестингом та живим трейдингом.

## 🚀 Основні можливості

- **SMC Аналіз**: Автоматичне виявлення Order Blocks, Fair Value Gaps, Liquidity Levels
- **Веб-панель**: Сучасний веб-інтерфейс для управління парами та моніторингу
- **Автоматичний бектест**: Перевірка стратегії перед живим трейдингом
- **Живий трейдинг**: Автоматичне виконання сигналів з управлінням ризиками
- **Telegram інтеграція**: Сповіщення про сигнали та статус торгівлі
- **Docker підтримка**: Легке розгортання та масштабування
- **Кешування даних**: Оптимізована робота з Binance API

## 🏗️ Архітектура проекту

```
trading-bot-smc/
├── 📁 src/                          # Основний код проекту
│   ├── 📁 core/                     # Ядро системи
│   │   ├── exchange_gateway.py      # Шлюз для бірж
│   │   ├── pair_manager.py          # Управління торговими парами
│   │   └── strategy.py              # Базова стратегія
│   ├── 📁 web/                      # Веб-інтерфейс
│   │   ├── app.py                   # FastAPI додаток
│   │   ├── templates/               # HTML шаблони
│   │   └── static/                  # CSS/JS файли
│   ├── 📁 backtest/                 # Бектестинг
│   │   └── runner.py                # Запуск бектестів
│   ├── smc_detector.py              # SMC алгоритми
│   ├── signal_generator.py          # Генерація сигналів
│   ├── live_trading.py              # Живий трейдинг
│   ├── live_monitor.py              # Моніторинг сигналів
│   ├── telegram_client.py           # Telegram інтеграція
│   └── data_downloader.py           # Завантаження даних
├── 📁 config/                       # Конфігурація
│   ├── pairs.yaml                   # Налаштування пар
│   └── models.py                    # Моделі конфігурації
├── 📁 data/                         # Дані для аналізу
├── 📁 backtest_results/             # Результати бектестів
├── 📁 signals_history/              # Історія сигналів
├── 📁 web/                          # Альтернативна веб-панель
├── start_web_panel.py               # Запуск веб-панелі
├── docker-compose.yml               # Docker конфігурація
├── requirements.txt                  # Python залежності
└── README.md                        # Документація
```

## 🛠️ Встановлення

### Вимоги
- Python 3.8+
- Docker (опціонально)
- Binance API ключі

### Локальне встановлення

1. **Клонування репозиторію**
```bash
git clone <repository-url>
cd trading-bot-smc
```

2. **Створення віртуального середовища**
```bash
python -m venv venv
```

3. **Активація середовища**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. **Встановлення залежностей**
```bash
pip install -r requirements.txt
```

5. **Налаштування змінних середовища**
```bash
cp dot.env.example .env
# Відредагуйте .env файл з вашими API ключами
```

### Docker встановлення

1. **Запуск через Docker Compose**
```bash
docker-compose up -d
```

2. **Або збірка та запуск**
```bash
docker build -t smc-trading-bot .
docker run -p 8000:8000 smc-trading-bot
```

## 🚀 Запуск

### Веб-панель (рекомендовано)

```bash
python start_web_panel.py
```

Веб-інтерфейс буде доступний за адресою: http://localhost:8000

**Можливості веб-панелі:**
- ✅ Управління торговими парами
- ✅ Перемикачі бектесту для кожної пари
- ✅ Контроль живого трейдингу
- ✅ Реальний час сигналів
- ✅ Гаряче перезавантаження конфігурації

### Консольний запуск

#### Основна торгова система
```bash
python main.py --ltf data/btc_15m.csv --htf data/btc_4h.csv
```

#### Живий трейдинг з бектестом
```bash
python live_trading.py --symbol BTCUSDT --run-backtest
```

#### Бектест окремо
```bash
python src/backtester.py --symbol BTCUSDT --days 30
```

### Batch файли (Windows)

```bash
# Запуск меню трейдингу
start_trading_menu.bat

# Запуск Docker
docker-start.bat
```

## ⚙️ Конфігурація

### Основні параметри

```python
# config.py
@dataclass
class SMCConfig:
    fractal_left: int = 2              # Ліві бари для фракталів
    fractal_right: int = 2             # Праві бари для фракталів
    min_risk_reward: float = 3.0       # Мінімальний RR
    max_risk_per_trade: float = 0.02   # Максимальний ризик на угоду
    premium_discount_lookback: int = 500  # Кандели для P/D розрахунку
    ob_lookback: int = 10              # Кандели для OB виявлення
```

### Налаштування пар

```yaml
# config/pairs.yaml
pairs:
  BTCUSDT:
    enabled: true
    backtest_enabled: true
    risk_percent: 2.0
    max_positions: 1
  ETHUSDT:
    enabled: true
    backtest_enabled: false
    risk_percent: 1.5
    max_positions: 2
```

### Змінні середовища

```bash
# .env
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 📊 Використання

### 1. Веб-панель

1. Запустіть `python start_web_panel.py`
2. Відкрийте http://localhost:8000
3. Налаштуйте торгові пари
4. Запустіть бектест для кожної пари
5. Активуйте живий трейдинг

### 2. Консольний режим

#### Бектест перед трейдингом
```bash
python live_trading.py --symbol SOLUSDT --run-backtest --backtest-days 60
```

#### Запуск без підтвердження
```bash
python live_trading.py --symbol ETHUSDT --run-backtest --skip-backtest-prompt
```

### 3. Завантаження даних

```bash
python download_data.py --symbol BTCUSDT --timeframe 15m --days 30
```

## 🔧 API Endpoints

### Веб-панель API

- `GET /` - Головна сторінка
- `GET /pairs` - Список пар
- `POST /pairs/{symbol}/toggle` - Перемикання пари
- `POST /pairs/{symbol}/backtest` - Запуск бектесту
- `GET /signals` - Список сигналів
- `POST /start-trading` - Запуск трейдингу

### Бектест API

- `POST /api/backtest/{symbol}` - Запуск бектесту
- `GET /api/backtest/{symbol}/results` - Результати бектесту
- `POST /api/start-with-backtest` - Запуск з бектестом

## 📈 SMC Алгоритми

### Order Block Detection
- Автоматичне виявлення Order Blocks на різних таймфреймах
- Фільтрація за Premium/Discount зонами
- Валідація через Fair Value Gaps

### Fair Value Gap (FVG)
- Виявлення FVG на основі фракталів
- Аналіз confluence з Order Blocks
- Фільтрація сигналів за FVG

### Liquidity Levels
- Визначення рівнів ліквідності
- Аналіз stop loss зон
- Оптимізація входів

## 🧪 Тестування

```bash
# Запуск всіх тестів
python -m pytest tests/

# Конкретний тест
python -m pytest tests/test_smc_detector.py

# З покриттям
python -m pytest --cov=src tests/
```

## 📊 Моніторинг

### Логи
- Структуровані логи через `structlog`
- Рівні: DEBUG, INFO, WARNING, ERROR
- Автоматична ротація файлів

### Telegram сповіщення
- Сповіщення про нові сигнали
- Статус торгівлі
- Результати бектестів

### Веб-моніторинг
- Реальний час статусу пар
- Графіки продуктивності
- Алерти та сповіщення

## 🚨 Безпека

- API ключі зберігаються в `.env` файлі
- Обмеження доступу до веб-панелі
- Валідація всіх вхідних даних
- Логування всіх операцій

## 🔄 Розробка

### Структура коду
- Модульна архітектура
- Типізація через type hints
- Документація коду
- Тести для всіх модулів

### Додавання нових пар
1. Додайте пару в `config/pairs.yaml`
2. Налаштуйте параметри ризику
3. Запустіть бектест
4. Активуйте в веб-панелі

### Створення нових стратегій
1. Успадкуйте базовий клас `Strategy`
2. Реалізуйте методи `generate_signals`
3. Додайте тести
4. Інтегруйте в систему

## 📚 Документація

- [SMC Concepts](https://www.smartmoneyconcepts.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Binance API Reference](https://binance-docs.github.io/apidocs/)

## 🤝 Внесок

1. Fork репозиторію
2. Створіть feature branch
3. Зробіть зміни
4. Додайте тести
5. Створіть Pull Request

## 📄 Ліцензія

Цей проект розповсюджується під MIT ліцензією.

## 🆘 Підтримка

- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Telegram**: [Bot Support](link-to-telegram)

## 🔮 Roadmap

- [ ] Мультибіржова підтримка
- [ ] Machine Learning інтеграція
- [ ] Мобільний додаток
- [ ] Розширена аналітика
- [ ] Соціальна торгівля

---

**⚠️ Дисклеймер**: Цей бот призначений тільки для освітніх цілей. Торгівля криптовалютами має високий ризик. Завжди робіть власне дослідження та не інвестуйте більше, ніж можете дозволити собі втратити.