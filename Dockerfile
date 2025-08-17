# Multi-stage Dockerfile for backtest and live monitor
# Base image
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies layer (build/cache Python deps)
FROM base AS deps
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt

# Runtime base: app code + virtualenv
FROM base AS runtime
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=deps /opt/venv /opt/venv
COPY . /app

# Final targets
FROM runtime AS backtest
CMD ["python", "main.py"]

FROM runtime AS live
CMD ["python", "live_trading.py"]

# Usage examples:
# Build backtest image:
#   docker build --target backtest -t smc-bot:backtest .
# Run backtest (mount data folder with CSVs):
#   docker run --rm -it -v %cd%/data:/app/data smc-bot:backtest \
#     python main.py --ltf data/btc_15m.csv --htf data/btc_4h.csv
#
# Build live monitor image:
#   docker build --target live -t smc-bot:live .
# Run live (pass tokens via env or CLI):
#   docker run --rm -it \
#     -e TELEGRAM_TOKEN=xxx -e TELEGRAM_CHAT_ID=yyy \
#     smc-bot:live python live_trading.py --symbol ETHUSDT --desktop-alerts

