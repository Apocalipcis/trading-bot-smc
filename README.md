# SMC Trading Bot

Smart Money Concepts торговий бот для аналізу фінансових ринків.

## Структура проекту

```
trading-bot-smc/
├── src/                    # Основний код
│   ├── models.py          # Data класи (Swing, OB, FVG, Signal)
│   ├── smc_detector.py    # SMC алгоритми виявлення
│   ├── signal_generator.py # Генерація торгових сигналів
│   └── data_loader.py     # Робота з CSV файлами
├── data/                  # CSV файли з даними
├── config.py              # Налаштування бота
├── main.py                # Головний файл запуску
├── requirements.txt       # Python залежності
└── README.md             # Цей файл
```

## Встановлення

1. Створити віртуальне середовище:
```bash
python -m venv venv
```

2. Активувати (Windows):
```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Встановити залежності:
```bash
venv\Scripts\python.exe -m pip install pandas numpy
```

## Використання

### Запуск бектесту

```bash
venv\Scripts\python.exe main.py --ltf data/btc_15m.csv --htf data/btc_4h.csv
```

### Параметри командного рядка

- `--ltf` - Lower timeframe CSV файл (обов'язково)
- `--htf` - Higher timeframe CSV файл (обов'язково)
- `--rr` - Мінімальний Risk/Reward ratio (за замовчуванням: 2.0)
- `--left` - Fractal left bars (за замовчуванням: 2)
- `--right` - Fractal right bars (за замовчуванням: 2)
- `--out` - Вихідний файл (за замовчуванням: signals.csv)
- `--require-fvg` - Вимагати FVG confluence для сигналів

### Приклади

```bash
# Базовий запуск
venv\Scripts\python.exe main.py --ltf data/btc_15m.csv --htf data/btc_4h.csv

# З кастомними параметрами
venv\Scripts\python.exe main.py --ltf data/btc_15m.csv --htf data/btc_4h.csv --rr 3.0 --out my_signals.csv

# З обов'язковим FVG confluence
venv\Scripts\python.exe main.py --ltf data/btc_15m.csv --htf data/btc_4h.csv --require-fvg
```

## Формат CSV файлів

CSV файли повинні містити колонки:
```
timestamp,open,high,low,close,volume
```

Timestamp може бути:
- Unix timestamp в мілісекундах
- ISO8601 формат (2024-01-01 00:00:00)

## SMC концепції

Бот використовує наступні Smart Money Concepts:

- **Fractal Pivots** - Виявлення swing high/low точок
- **BOS/CHOCH** - Break of Structure / Change of Character
- **Order Blocks** - Зони інституційних ордерів
- **Fair Value Gaps (FVG)** - Ціновi розриви
- **Premium/Discount** - Позиція ціни відносно діапазону

## Результати

Результати зберігаються в CSV файл з колонками:
- `timestamp` - Час входу
- `direction` - LONG/SHORT
- `entry` - Ціна входу
- `sl` - Stop Loss
- `tp` - Take Profit
- `rr` - Risk/Reward ratio
- `htf_bias` - HTF тренд (bull/bear)
- `fvg_confluence` - Чи є FVG confluence

## Наступні кроки

Для покращення бота можна додати:
- Підключення до API біржі для живої торгівлі
- Систему бектестингу з детальною статистикою
- Візуалізацію сигналів та графіків
- Більш складний risk management
- Уведомлення в Telegram