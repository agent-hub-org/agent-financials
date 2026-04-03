"""
Chart data builder for the /charts/{ticker} endpoint.

Fetches OHLCV history from yfinance and computes technical indicators
using pandas (already a yfinance dependency — no new dep required).
Returns serializable dicts; no LLM involved.
"""

from __future__ import annotations

import logging
import math

import yfinance as yf
from cachetools import cached, TTLCache

logger = logging.getLogger("agent_financials.charts")

# Periods the frontend can request
VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}
VALID_INTERVALS = {"1d", "1wk"}


def _ema_series(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average. Returns None for the first `period-1` positions."""
    result: list[float | None] = [None] * (period - 1)
    if len(values) < period:
        return [None] * len(values)
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    result.append(round(ema, 4))
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
        result.append(round(ema, 4))
    return result


def _rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """RSI using Wilder smoothing. Returns None for first `period` positions."""
    result: list[float | None] = [None] * period
    if len(closes) <= period:
        return [None] * len(closes)
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0)) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        rsi_val = 100 - 100 / (1 + rs)
        result.append(round(rsi_val, 2))
    return result


def _bollinger(closes: list[float], period: int = 20, num_std: float = 2.0) -> tuple[
    list[float | None], list[float | None], list[float | None]
]:
    """Returns (upper, middle, lower) Bollinger Band series."""
    upper: list[float | None] = []
    middle: list[float | None] = []
    lower: list[float | None] = []
    for i in range(len(closes)):
        if i < period - 1:
            upper.append(None)
            middle.append(None)
            lower.append(None)
        else:
            window = closes[i - period + 1: i + 1]
            sma = sum(window) / period
            std = math.sqrt(sum((v - sma) ** 2 for v in window) / period)
            upper.append(round(sma + num_std * std, 2))
            middle.append(round(sma, 2))
            lower.append(round(sma - num_std * std, 2))
    return upper, middle, lower


def _sma(closes: list[float], period: int) -> list[float | None]:
    result: list[float | None] = []
    for i in range(len(closes)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(round(sum(closes[i - period + 1: i + 1]) / period, 2))
    return result


@cached(cache=TTLCache(maxsize=100, ttl=900))
def fetch_chart_data(ticker: str, period: str = "1y") -> dict:
    """
    Fetch OHLCV + technical indicators for `ticker` over `period`.
    Returns a serializable dict ready for JSON response.
    Raises ValueError for invalid inputs, RuntimeError if yfinance fails.
    """
    period = period if period in VALID_PERIODS else "1y"

    logger.info("Building chart data for ticker='%s', period='%s'", ticker, period)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval="1d")
    except Exception as e:
        raise RuntimeError(f"yfinance fetch failed for {ticker}: {e}") from e

    if hist.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'.")

    # Normalize timezone-aware DatetimeIndex to date strings
    hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
    dates = hist.index.strftime("%Y-%m-%d").tolist()

    opens = [round(float(v), 2) for v in hist["Open"]]
    highs = [round(float(v), 2) for v in hist["High"]]
    lows = [round(float(v), 2) for v in hist["Low"]]
    closes = [round(float(v), 2) for v in hist["Close"]]
    volumes = [int(v) for v in hist["Volume"]]

    # Technical indicators
    sma_20 = _sma(closes, 20)
    sma_50 = _sma(closes, 50)
    sma_200 = _sma(closes, 200)
    rsi_14 = _rsi(closes, 14)

    ema_12 = _ema_series(closes, 12)
    ema_26 = _ema_series(closes, 26)
    macd_line = [
        round(e12 - e26, 4) if e12 is not None and e26 is not None else None
        for e12, e26 in zip(ema_12, ema_26)
    ]
    macd_values = [v for v in macd_line if v is not None]
    macd_signal_raw = _ema_series(macd_values, 9)
    # Pad signal to full length
    pad = len(macd_line) - len(macd_signal_raw)
    macd_signal = [None] * pad + macd_signal_raw
    macd_histogram = [
        round(m - s, 4) if m is not None and s is not None else None
        for m, s in zip(macd_line, macd_signal)
    ]

    bb_upper, bb_middle, bb_lower = _bollinger(closes, 20, 2.0)

    # Summary metrics
    current_price = closes[-1] if closes else None
    high_52w = max(highs) if highs else None
    low_52w = min(lows) if lows else None
    start_price = closes[0] if closes else None
    ytd_return = round((current_price / start_price - 1) * 100, 2) if start_price and current_price else None

    # Annualized volatility from daily returns
    ann_vol = None
    if len(closes) >= 20:
        daily_rets = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
        mean_ret = sum(daily_rets) / len(daily_rets)
        variance = sum((r - mean_ret) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
        ann_vol = round(math.sqrt(variance) * math.sqrt(252) * 100, 2)

    avg_vol_10d = int(sum(volumes[-10:]) / min(10, len(volumes))) if volumes else None
    avg_vol_30d = int(sum(volumes[-30:]) / min(30, len(volumes))) if volumes else None

    # Fetch company name
    try:
        info = t.fast_info
        company_name = t.info.get("shortName") or ticker
    except Exception:
        company_name = ticker

    return {
        "ticker": ticker,
        "name": company_name,
        "period": period,
        "data_points": len(dates),
        "ohlcv": [
            {"date": d, "open": o, "high": h, "low": l, "close": c, "volume": v}
            for d, o, h, l, c, v in zip(dates, opens, highs, lows, closes, volumes)
        ],
        "indicators": {
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi_14": rsi_14,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_histogram,
            "bollinger_upper": bb_upper,
            "bollinger_middle": bb_middle,
            "bollinger_lower": bb_lower,
        },
        "summary": {
            "current_price": current_price,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "period_return_pct": ytd_return,
            "annualized_volatility_pct": ann_vol,
            "avg_volume_10d": avg_vol_10d,
            "avg_volume_30d": avg_vol_30d,
        },
    }
