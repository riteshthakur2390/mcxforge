# =============================================================
# MCXForge — Dockerfile
# MCX Commodity Futures Algorithmic Trading Platform
# =============================================================

FROM --platform=linux/amd64 python:3.11-slim

LABEL maintainer="MCXForge"
LABEL description="MCX Commodity Futures Algorithmic Engine"

# ── System deps only (no TA-Lib compile) ─────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── PYTHONPATH ────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONWARNINGS="ignore::ImportWarning"

# ── Install Python dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir ta-lib-bin

# Just copy the source files — no install needed
COPY pandas_ta /app/pandas_ta

# ── Copy application code ─────────────────────────────────────
COPY . .

# ── Create persistent directories ─────────────────────────────
RUN mkdir -p data/cache data/historical journal logs \
             ml/saved_models backtesting/results

# ── Expose dashboard port ─────────────────────────────────────
EXPOSE 5054

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5054/ || exit 1

# ── Start MCXForge ────────────────────────────────────────────
CMD ["python", "main.py"]
