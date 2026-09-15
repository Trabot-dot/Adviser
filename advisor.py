#!/usr/bin/env python3
"""
Market Advisor: a daily research brief, not a trading bot.

Each run it:
  1. downloads daily prices for the long-term stock list and the short-term crypto list
     (Alpaca market data, which is free with a paper account's keys),
  2. pulls the last few days of news for every symbol (Alpaca News),
  3. works out trend, momentum and volatility numbers for each,
  4. asks an AI model to weigh the numbers and the news together and write the brief:
     picks with reasons, a review of your holdings, news flashes and a to-do list,
  5. raises alerts (and opens a GitHub issue, which emails you) when something needs doing,
  6. writes docs/data/latest.json for the web app, plus a history of past briefs.

It never places trades. The AI can only rate symbols it was given and every number
it quotes is checked against the data. If the AI is unavailable, a rules-only brief
is produced instead so the app always has something to show.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
import tomllib
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
ROOT = pathlib.Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DATA = DOCS / "data"
TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class AdvisorError(Exception):
    """A problem worth stopping for, with a plain-English message."""


def log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

GLOBAL_DEFAULTS = {
    "benchmark": "SPY", "data_feed": "sip", "news_days": 3, "max_long_picks": 8, "max_short_picks": 4,
    "ai_provider": "auto", "ai_model_github": "openai/gpt-4.1", "ai_model_anthropic": "claude-sonnet-4-5",
}
SLEEVE_DEFAULTS = {
    "ma_fast": 50, "ma_slow": 200, "rsi_period": 14, "atr_period": 14,
    "stop_pct": 10.0, "target_pct": 0.0, "max_hold_days": 0, "symbols": [],
}
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}(/USD)?$")


def load_config(path: pathlib.Path) -> dict:
    raw = {}
    if path.exists():
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise AdvisorError(f"config.toml has a typo and can't be read: {e}") from e
    cfg = dict(GLOBAL_DEFAULTS)
    for k, v in raw.items():
        if not isinstance(v, dict):
            cfg[k] = v
    cfg["benchmark"] = str(cfg["benchmark"]).upper().strip()
    if cfg["data_feed"] not in ("sip", "iex"):
        raise AdvisorError('config.toml: data_feed must be "sip" or "iex".')
    if cfg["ai_provider"] not in ("auto", "github", "anthropic", "none"):
        raise AdvisorError('config.toml: ai_provider must be "auto", "github", "anthropic" or "none".')
    for k in ("news_days", "max_long_picks", "max_short_picks"):
        if not isinstance(cfg[k], int) or cfg[k] < 0:
            raise AdvisorError(f"config.toml: {k} must be a whole number.")
    sleeves = {}
    for name in ("long", "short"):
        s = dict(SLEEVE_DEFAULTS)
        s.update(raw.get(name, {}) if isinstance(raw.get(name), dict) else {})
        for k in ("ma_fast", "ma_slow", "rsi_period", "atr_period", "max_hold_days"):
            if not isinstance(s[k], int) or s[k] < 0:
                raise AdvisorError(f"config.toml: [{name}] {k} must be a whole number.")
        for k in ("stop_pct", "target_pct"):
            if not isinstance(s[k], (int, float)) or s[k] < 0:
                raise AdvisorError(f"config.toml: [{name}] {k} must be a number.")
            s[k] = float(s[k])
        if s["ma_slow"] <= s["ma_fast"]:
            raise AdvisorError(f"config.toml: [{name}] ma_slow must be bigger than ma_fast.")
        syms = []
        for x in s["symbols"] or []:
            sym = str(x).strip().upper()
            if not SYMBOL_RE.match(sym):
                raise AdvisorError(f"config.toml: '{x}' in [{name}] doesn't look like a ticker (stocks like AAPL, crypto like BTC/USD).")
            if sym not in syms:
                syms.append(sym)
        s["symbols"] = syms
        sleeves[name] = s
    if not sleeves["long"]["symbols"] and not sleeves["short"]["symbols"]:
        raise AdvisorError("config.toml needs some symbols under [long] or [short].")
    cfg["sleeves"] = sleeves
    return cfg


def load_portfolio(path: pathlib.Path, cfg: dict) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AdvisorError(f"portfolio.json has a typo and can't be read: {e}") from e
    out = []
    for h in raw.get("holdings", []) if isinstance(raw, dict) else []:
        if not isinstance(h, dict):
            continue
        sym = str(h.get("symbol", "")).strip().upper()
        if not SYMBOL_RE.match(sym):
            continue
        sleeve = h.get("sleeve") if h.get("sleeve") in ("long", "short") else ("short" if "/" in sym else "long")
        try:
            qty = float(h.get("qty", 0) or 0)
            entry = float(h.get("entry_price", 0) or 0)
        except (TypeError, ValueError):
            qty, entry = 0.0, 0.0
        entry_date = str(h.get("entry_date", "") or "")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", entry_date):
            entry_date = ""
        out.append({
            "symbol": sym, "sleeve": sleeve, "qty": qty, "entry_price": entry, "entry_date": entry_date,
            "stop_pct": float(h["stop_pct"]) if isinstance(h.get("stop_pct"), (int, float)) else cfg["sleeves"][sleeve]["stop_pct"],
            "target_pct": float(h["target_pct"]) if isinstance(h.get("target_pct"), (int, float)) else cfg["sleeves"][sleeve]["target_pct"],
            "note": str(h.get("note", "") or "")[:200],
        })
    return out


# ---------------------------------------------------------------------------
# Alpaca market data (prices + news)
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})?$")


def parse_ts(value: str) -> datetime:
    m = _TS_RE.match(str(value).strip())
    if not m:
        raise ValueError(f"bad timestamp {value!r}")
    frac = (m.group(2) or "")[:7]
    tz = m.group(3) or "Z"
    return datetime.fromisoformat(m.group(1) + frac + ("+00:00" if tz == "Z" else tz))


class FeedNotAllowed(Exception):
    pass


class Alpaca:
    def __init__(self, key_id: str, secret: str, trading_url: str = TRADING_URL, data_url: str = DATA_URL):
        self.trading_url = trading_url.rstrip("/")
        self.data_url = data_url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"})

    def _get(self, url: str, params=None, attempts: int = 4):
        last = None
        for i in range(attempts):
            try:
                r = self.s.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                last = f"network problem: {e}"
                time.sleep(2 ** i)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"Alpaca returned {r.status_code}"
                time.sleep(min(30, 3 * 2 ** i))
                continue
            return r
        raise AdvisorError(f"Couldn't reach Alpaca after {attempts} tries ({last}).")

    def _json(self, r, what: str):
        if r.status_code in (401, 403):
            raise AdvisorError(f"Alpaca refused the request for {what} ({r.status_code}). Check the GitHub secrets ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.")
        if not r.ok:
            raise AdvisorError(f"Alpaca error getting {what} ({r.status_code}): {r.text[:300]}")
        return r.json()

    def clock(self) -> dict:
        return self._json(self._get(f"{self.trading_url}/v2/clock"), "the market clock")

    def stock_bars(self, symbols: list[str], start: date, end_utc: datetime, feed: str) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {s: [] for s in symbols}
        for i in range(0, len(symbols), 40):
            group = symbols[i:i + 40]
            params = {"symbols": ",".join(group), "timeframe": "1Day", "start": start.isoformat(),
                      "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "adjustment": "all", "feed": feed, "limit": 10000}
            while True:
                r = self._get(f"{self.data_url}/v2/stocks/bars", params)
                if r.status_code in (401, 403, 422) and feed == "sip":
                    raise FeedNotAllowed(r.text[:200])
                data = self._json(r, "stock prices")
                self._collect(out, data.get("bars") or {})
                if not data.get("next_page_token"):
                    break
                params["page_token"] = data["next_page_token"]
        return self._finish(out)

    def crypto_bars(self, symbols: list[str], start: date) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {s: [] for s in symbols}
        if not symbols:
            return out
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start.isoformat(), "limit": 10000}
        while True:
            data = self._json(self._get(f"{self.data_url}/v1beta3/crypto/us/bars", params), "crypto prices")
            self._collect(out, data.get("bars") or {})
            if not data.get("next_page_token"):
                break
            params["page_token"] = data["next_page_token"]
        return self._finish(out)

    @staticmethod
    def _collect(out, bars):
        for sym, rows in bars.items():
            for b in rows or []:
                try:
                    out.setdefault(sym, []).append({
                        "d": parse_ts(b["t"]).astimezone(ET).date().isoformat(),
                        "o": float(b["o"]), "h": float(b["h"]), "l": float(b["l"]), "c": float(b["c"]), "v": float(b.get("v", 0) or 0),
                    })
                except (KeyError, ValueError, TypeError):
                    continue

    @staticmethod
    def _finish(out):
        for sym in out:
            dedup = {b["d"]: b for b in out[sym]}
            out[sym] = [dedup[k] for k in sorted(dedup)]
        return out

    def latest_stock_trades(self, symbols: list[str]) -> dict[str, float]:
        out = {}
        for i in range(0, len(symbols), 100):
            r = self._get(f"{self.data_url}/v2/stocks/trades/latest", {"symbols": ",".join(symbols[i:i + 100]), "feed": "iex"})
            if not r.ok:
                continue
            for sym, t in (r.json().get("trades") or {}).items():
                try:
                    out[sym] = float(t["p"])
                except (KeyError, ValueError, TypeError):
                    pass
        return out

    def latest_crypto_trades(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        r = self._get(f"{self.data_url}/v1beta3/crypto/us/latest/trades", {"symbols": ",".join(symbols)})
        if not r.ok:
            return {}
        out = {}
        for sym, t in (r.json().get("trades") or {}).items():
            try:
                out[sym] = float(t["p"])
            except (KeyError, ValueError, TypeError):
                pass
        return out

    def news(self, symbols: list[str], start_utc: datetime, per_symbol: int = 6) -> dict[str, list[dict]]:
        """Recent headlines per symbol. Crypto symbols use Alpaca's BTCUSD form."""
        out: dict[str, list[dict]] = {s: [] for s in symbols}
        alias = {s.replace("/", ""): s for s in symbols}
        for i in range(0, len(symbols), 20):
            group = symbols[i:i + 20]
            params = {"symbols": ",".join(s.replace("/", "") for s in group), "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "limit": 50, "sort": "desc", "include_content": "false"}
            for _ in range(4):  # up to 4 pages per group
                r = self._get(f"{self.data_url}/v1beta1/news", params)
                if not r.ok:
                    log(f"News unavailable ({r.status_code}): {r.text[:120]}")
                    return out
                data = r.json()
                for a in data.get("news") or []:
                    item = {
                        "headline": str(a.get("headline", ""))[:200], "source": str(a.get("source", ""))[:40],
                        "url": str(a.get("url", ""))[:300], "at": str(a.get("created_at", a.get("updated_at", "")))[:19].replace("T", " "),
                        "summary": re.sub(r"\s+", " ", str(a.get("summary", "") or ""))[:240],
                    }
                    for s in a.get("symbols") or []:
                        real = alias.get(str(s).upper())
                        if real and len(out[real]) < per_symbol and not any(x["headline"] == item["headline"] for x in out[real]):
                            out[real].append(dict(item))
                if not data.get("next_page_token"):
                    break
                params["page_token"] = data["next_page_token"]
        return out


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def sma(values, n):
    out = [None] * len(values)
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= n:
            total -= values[i - n]
        if i >= n - 1:
            out[i] = total / n
    return out


def rsi_wilder(closes, n):
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    gain = loss = 0.0
    for i in range(1, n + 1):
        ch = closes[i] - closes[i - 1]
        gain += max(ch, 0)
        loss += max(-ch, 0)
    ag, al = gain / n, loss / n

    def calc():
        if al == 0:
            return 50.0 if ag == 0 else 100.0
        return 100 - 100 / (1 + ag / al)

    out[n] = calc()
    for i in range(n + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(ch, 0)) / n
        al = (al * (n - 1) + max(-ch, 0)) / n
        out[i] = calc()
    return out


def atr_wilder(bars, n):
    out = [None] * len(bars)
    if len(bars) <= n:
        return out
    tr = []
    for i, b in enumerate(bars):
        if i == 0:
            tr.append(b["h"] - b["l"])
        else:
            pc = bars[i - 1]["c"]
            tr.append(max(b["h"] - b["l"], abs(b["h"] - pc), abs(b["l"] - pc)))
    a = sum(tr[1:n + 1]) / n
    out[n] = a
    for i in range(n + 1, len(bars)):
        a = (a * (n - 1) + tr[i]) / n
        out[i] = a
    return out


def pct(a, b):
    return (a / b - 1) * 100 if a is not None and b else None


@dataclass
class Snapshot:
    symbol: str
    sleeve: str
    ok: bool = False
    price: float | None = None
    day_pct: float | None = None
    week_pct: float | None = None
    month_pct: float | None = None
    trend: str | None = None       # up | down
    fast: float | None = None
    slow: float | None = None
    rsi: float | None = None
    atr_pct: float | None = None   # ATR as % of price
    high_52w: float | None = None
    off_high_pct: float | None = None
    cross: str | None = None       # golden | death | None (last 5 bars)
    setup: str = ""                # plain-English rules read
    score: float = 0.0             # ranking score for candidate selection
    bars: int = 0
    news: list = field(default_factory=list)
    note: str = ""


def snapshot(symbol: str, sleeve: str, bars: list[dict], s: dict) -> Snapshot:
    snap = Snapshot(symbol, sleeve, bars=len(bars))
    if len(bars) < 2:
        snap.note = "no price data"
        return snap
    closes = [b["c"] for b in bars]
    i = len(closes) - 1
    snap.price = closes[i]
    snap.day_pct = pct(closes[i], closes[i - 1])
    snap.week_pct = pct(closes[i], closes[i - 6]) if i >= 6 else None
    snap.month_pct = pct(closes[i], closes[i - 22]) if i >= 22 else None
    hi = max(b["h"] for b in bars[-252:])
    snap.high_52w = hi
    snap.off_high_pct = pct(closes[i], hi)
    fast_a, slow_a = sma(closes, s["ma_fast"]), sma(closes, s["ma_slow"])
    rsi_a, atr_a = rsi_wilder(closes, s["rsi_period"]), atr_wilder(bars, s["atr_period"])
    snap.fast, snap.slow, snap.rsi = fast_a[i], slow_a[i], rsi_a[i]
    if atr_a[i] and snap.price:
        snap.atr_pct = atr_a[i] / snap.price * 100
    if snap.slow is None or snap.rsi is None:
        snap.note = f"needs {max(s['ma_slow'], s['rsi_period'] + 1)} days of history (has {len(bars)})"
        return snap
    snap.ok = True
    snap.trend = "up" if snap.fast > snap.slow else "down"
    for k in range(i, max(i - 5, 0), -1):
        if fast_a[k - 1] is None or slow_a[k - 1] is None:
            break
        prev, cur = fast_a[k - 1] - slow_a[k - 1], fast_a[k] - slow_a[k]
        if prev <= 0 < cur:
            snap.cross = "golden"
            break
        if prev >= 0 > cur:
            snap.cross = "death"
            break
    r = snap.rsi
    if snap.cross == "golden":
        snap.setup, snap.score = f"golden cross: {s['ma_fast']}-day average just moved above the {s['ma_slow']}-day", 90
    elif snap.cross == "death":
        snap.setup, snap.score = f"death cross: {s['ma_fast']}-day average just fell below the {s['ma_slow']}-day", 5
    elif snap.trend == "up" and r <= 45:
        snap.setup, snap.score = f"pullback in an uptrend (RSI {r:.0f})", 80 + (45 - r)
    elif snap.trend == "up" and r >= 70:
        snap.setup, snap.score = f"uptrend but overbought (RSI {r:.0f})", 40
    elif snap.trend == "up":
        snap.setup, snap.score = f"steady uptrend (RSI {r:.0f})", 60 + (60 - r) / 4
    elif snap.trend == "down" and r <= 30:
        snap.setup, snap.score = f"downtrend, oversold (RSI {r:.0f}), possible bounce only", 25
    elif snap.trend == "down" and r >= 60:
        snap.setup, snap.score = f"bounce in a downtrend (RSI {r:.0f})", 10
    else:
        snap.setup, snap.score = f"downtrend (RSI {r:.0f})", 15
    if snap.month_pct is not None:
        snap.score += max(-10, min(10, snap.month_pct / 2))
    return snap


# ---------------------------------------------------------------------------
# Holdings, alerts and rules-only reasoning
# ---------------------------------------------------------------------------

NEGATIVE_WORDS = ["downgrade", "cuts guidance", "lowers guidance", "misses", "investigation", "lawsuit", "recall", "sec ", "probe",
                  "halted", "bankrupt", "layoffs", "resigns", "steps down", "delay", "warning", "plunge", "tumble", "slump", "sanction"]
POSITIVE_WORDS = ["upgrade", "raises guidance", "beats", "record", "contract", "award", "wins", "approval", "buyback", "dividend increase",
                  "partnership", "surge", "rally", "soar", "outperform", "backlog"]


def headline_tone(text: str) -> str:
    t = text.lower()
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    return "negative" if neg > pos else "positive" if pos > neg else "neutral"


def review_holding(h: dict, snap: Snapshot | None, today: str) -> dict:
    out = {**h, "price": snap.price if snap else None, "pnl_pct": None, "value": None, "days_held": None,
           "stop": None, "target": None, "advice": "hold", "reasons": [], "alerts": [], "trend": snap.trend if snap else None, "rsi": snap.rsi if snap else None}
    if not snap or not snap.price:
        out["advice"] = "check"
        out["reasons"].append("No price data for this symbol. Check the ticker in portfolio.json.")
        return out
    if h["entry_price"] > 0:
        out["pnl_pct"] = pct(snap.price, h["entry_price"])
        out["stop"] = h["entry_price"] * (1 - h["stop_pct"] / 100) if h["stop_pct"] > 0 else None
        out["target"] = h["entry_price"] * (1 + h["target_pct"] / 100) if h["target_pct"] > 0 else None
    if h["qty"] > 0:
        out["value"] = h["qty"] * snap.price
    if h["entry_date"]:
        try:
            out["days_held"] = (date.fromisoformat(today) - date.fromisoformat(h["entry_date"])).days
        except ValueError:
            pass
    reasons, alerts = out["reasons"], out["alerts"]
    if out["stop"] is not None and snap.price <= out["stop"]:
        out["advice"] = "sell"
        reasons.append(f"Price {snap.price:.2f} is at or below your stop-loss of {out['stop']:.2f} ({out['pnl_pct']:+.1f}%).")
        alerts.append({"severity": "high", "title": f"{h['symbol']}: stop-loss hit", "detail": reasons[-1], "action": "Consider selling to cap the loss."})
    elif out["target"] is not None and snap.price >= out["target"]:
        out["advice"] = "trim"
        reasons.append(f"Price {snap.price:.2f} has reached your target of {out['target']:.2f} ({out['pnl_pct']:+.1f}%).")
        alerts.append({"severity": "high", "title": f"{h['symbol']}: profit target reached", "detail": reasons[-1], "action": "Consider taking some or all profit."})
    elif h["sleeve"] == "short" and h.get("max_hold_days", 0) and out["days_held"] is not None and out["days_held"] >= h["max_hold_days"]:
        out["advice"] = "sell"
        reasons.append(f"Held {out['days_held']} days, past the {h['max_hold_days']}-day limit for a short-term trade.")
        alerts.append({"severity": "medium", "title": f"{h['symbol']}: time limit reached", "detail": reasons[-1], "action": "Consider closing the trade."})
    if snap.ok and snap.cross == "death":
        if out["advice"] == "hold":
            out["advice"] = "sell" if h["sleeve"] == "short" else "review"
        reasons.append("Trend has broken: " + snap.setup + ".")
        alerts.append({"severity": "medium", "title": f"{h['symbol']}: trend break", "detail": snap.setup,
                       "action": "Consider closing the trade." if h["sleeve"] == "short" else "Review whether the long-term case still holds."})
    if snap.day_pct is not None and abs(snap.day_pct) >= 5:
        reasons.append(f"Big move today: {snap.day_pct:+.1f}%.")
        alerts.append({"severity": "medium", "title": f"{h['symbol']}: moved {snap.day_pct:+.1f}% today", "detail": "A move this size usually has a news cause. Check the flashes.", "action": "Check the news before reacting."})
    bad_news = [n for n in snap.news if headline_tone(n["headline"]) == "negative"]
    if bad_news:
        reasons.append(f"Negative headline: {bad_news[0]['headline']}")
        alerts.append({"severity": "medium", "title": f"{h['symbol']}: negative news", "detail": bad_news[0]["headline"], "action": "Read the story and decide whether it changes the case."})
    if not reasons:
        reasons.append(f"Nothing needs doing. {snap.setup[0].upper() + snap.setup[1:]}." if snap.setup else "Nothing needs doing.")
        if out["stop"] is not None:
            reasons.append(f"Stop-loss to watch: {out['stop']:.2f}.")
    for a in alerts:
        a["symbol"] = h["symbol"]
    return out


def rules_pick(snap: Snapshot, s: dict, sleeve: str) -> dict:
    """A pick written from the numbers alone (used when the AI is unavailable)."""
    action = "buy" if snap.score >= 75 else "watch" if snap.score >= 45 else "avoid"
    reasons = [snap.setup[0].upper() + snap.setup[1:] + "."]
    if snap.month_pct is not None:
        reasons.append(f"{snap.month_pct:+.1f}% over the last month, {snap.off_high_pct:+.1f}% from its 52-week high.")
    if snap.atr_pct:
        reasons.append(f"Typical daily move about {snap.atr_pct:.1f}%.")
    for n in snap.news[:2]:
        reasons.append(f"News: {n['headline']}")
    return {
        "symbol": snap.symbol, "sleeve": sleeve, "action": action, "conviction": 4 if action == "buy" else 2 if action == "watch" else 1,
        "price": snap.price, "horizon": "months" if sleeve == "long" else "days",
        "entry_note": "Buy on a normal day; avoid chasing a gap up." if action == "buy" else "Wait for a better setup." if action == "watch" else "Not now.",
        "stop": snap.price * (1 - s["stop_pct"] / 100) if snap.price else None,
        "target": snap.price * (1 + s["target_pct"] / 100) if snap.price and s["target_pct"] else None,
        "reasons": reasons, "risks": ["Rules-only view: the AI brief was unavailable, so the news has not been weighed."],
        "news": snap.news[:3], "indicators": indicators_dict(snap),
    }


def indicators_dict(snap: Snapshot) -> dict:
    return {"trend": snap.trend, "rsi": round(snap.rsi, 1) if snap.rsi is not None else None, "setup": snap.setup,
            "day_pct": round(snap.day_pct, 2) if snap.day_pct is not None else None,
            "week_pct": round(snap.week_pct, 2) if snap.week_pct is not None else None,
            "month_pct": round(snap.month_pct, 2) if snap.month_pct is not None else None,
            "off_high_pct": round(snap.off_high_pct, 1) if snap.off_high_pct is not None else None,
            "atr_pct": round(snap.atr_pct, 2) if snap.atr_pct is not None else None}


# ---------------------------------------------------------------------------
# AI brief
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful investment research analyst writing a daily brief for one private individual who trades a small account. You weigh recent news, the macro backdrop and the technical numbers together. You are direct, specific and honest about uncertainty. You never invent facts: every claim about news must come from the headlines provided, and every number from the data provided. You only rate the symbols you are given. This is research for the reader's own decisions, not personalised financial advice, and you say so once in the summary. Respond with valid JSON only, no markdown, no commentary outside the JSON."""


def compact_snapshot(snap: Snapshot, max_news: int = 3) -> dict:
    d = {"symbol": snap.symbol, "price": round(snap.price, 2) if snap.price else None, "trend": snap.trend,
         "setup": snap.setup, "rsi": round(snap.rsi) if snap.rsi is not None else None,
         "d1": round(snap.day_pct, 1) if snap.day_pct is not None else None, "w1": round(snap.week_pct, 1) if snap.week_pct is not None else None,
         "m1": round(snap.month_pct, 1) if snap.month_pct is not None else None, "off_high": round(snap.off_high_pct, 1) if snap.off_high_pct is not None else None,
         "vol_pct": round(snap.atr_pct, 1) if snap.atr_pct is not None else None,
         "news": [f"{n['at'][:10]} {n['headline']}" for n in snap.news[:max_news]]}
    return d


class AI:
    def __init__(self, cfg: dict):
        self.provider = None
        self.model = None
        want = cfg["ai_provider"]
        gh = os.environ.get("GITHUB_TOKEN", "").strip()
        an = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if want == "none":
            return
        if want in ("auto", "anthropic") and an:
            self.provider, self.model, self.key = "anthropic", cfg["ai_model_anthropic"], an
        elif want in ("auto", "github") and gh:
            self.provider, self.model, self.key = "github", cfg["ai_model_github"], gh
        self.github_url = os.environ.get("GITHUB_MODELS_URL", GITHUB_MODELS_URL)
        self.anthropic_url = os.environ.get("ANTHROPIC_URL", ANTHROPIC_URL)

    def ask(self, user_prompt: str, max_tokens: int = 2500) -> dict:
        if not self.provider:
            raise AdvisorError("No AI provider available (no GITHUB_TOKEN or ANTHROPIC_API_KEY).")
        for attempt in range(3):
            try:
                if self.provider == "github":
                    r = requests.post(self.github_url, timeout=120,
                                      headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "Accept": "application/json"},
                                      json={"model": self.model, "temperature": 0.2, "max_tokens": max_tokens,
                                            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]})
                    if r.status_code == 429:
                        time.sleep(20 * (attempt + 1))
                        continue
                    if not r.ok:
                        raise AdvisorError(f"AI request failed ({r.status_code}): {r.text[:200]}")
                    text = r.json()["choices"][0]["message"]["content"]
                else:
                    r = requests.post(self.anthropic_url, timeout=120,
                                      headers={"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                                      json={"model": self.model, "max_tokens": max_tokens, "temperature": 0.2, "system": SYSTEM_PROMPT,
                                            "messages": [{"role": "user", "content": user_prompt}]})
                    if r.status_code in (429, 529):
                        time.sleep(20 * (attempt + 1))
                        continue
                    if not r.ok:
                        raise AdvisorError(f"AI request failed ({r.status_code}): {r.text[:200]}")
                    text = "".join(p.get("text", "") for p in r.json().get("content", []))
                return parse_json(text)
            except requests.RequestException as e:
                if attempt == 2:
                    raise AdvisorError(f"Couldn't reach the AI service: {e}")
                time.sleep(5)
        raise AdvisorError("AI service is rate-limited right now.")


def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise AdvisorError("The AI reply wasn't valid JSON.")


def clean_list(v, n=6, maxlen=220):
    if not isinstance(v, list):
        return []
    return [str(x).strip()[:maxlen] for x in v if str(x).strip()][:n]


def ai_picks(ai: AI, sleeve: str, s: dict, cands: list[Snapshot], context: dict, max_picks: int) -> list[dict]:
    label = "long-term stock ideas to hold for months" if sleeve == "long" else "short-term crypto trades to hold for days"
    prompt = {
        "task": f"Choose up to {max_picks} {label} from the candidates, ranked best first. Also list any candidates that should be avoided right now and say why. For each pick give 3-5 bullet reasons that combine the news and the numbers, and 1-3 risks.",
        "today": context["today"], "market_context": context["market"],
        "rules_of_thumb": {
            "stop_pct": s["stop_pct"], "target_pct": s["target_pct"] or None,
            "guidance": ("Prefer uptrends with a healthy pullback or a fresh golden cross; be wary of buying after a big gap up; treat a death cross as a reason to avoid." if sleeve == "long"
                         else "Crypto trades 24/7 and moves fast: prefer coins with strong one-month momentum and a pullback, be explicit about the stop, and say 'watch' rather than 'buy' if the market is about to face a major event like a Fed decision."),
        },
        "candidates": [compact_snapshot(c) for c in cands],
        "output_schema": {
            "picks": [{"symbol": "string, must be a candidate", "action": "buy|watch", "conviction": "1-5", "horizon": "e.g. 'months' or '3-10 days'",
                       "entry_note": "one line on how/when to enter", "stop": "number, price", "target": "number, price, or null",
                       "reasons": ["3-5 bullets"], "risks": ["1-3 bullets"]}],
            "avoid": [{"symbol": "string", "why": "one line"}],
        },
    }
    data = ai.ask(json.dumps(prompt, separators=(",", ":")))
    by_sym = {c.symbol: c for c in cands}
    out = []
    for p in data.get("picks", []) if isinstance(data.get("picks"), list) else []:
        sym = str(p.get("symbol", "")).upper().strip()
        snap = by_sym.get(sym)
        if not snap or not snap.price:
            continue
        action = p.get("action") if p.get("action") in ("buy", "watch") else "watch"
        try:
            conviction = max(1, min(5, int(p.get("conviction", 3))))
        except (TypeError, ValueError):
            conviction = 3
        stop = to_float(p.get("stop"))
        target = to_float(p.get("target"))
        if stop is None or not (snap.price * 0.6 <= stop < snap.price):
            stop = snap.price * (1 - s["stop_pct"] / 100)
        if target is not None and not (snap.price < target <= snap.price * 2):
            target = None
        if target is None and s["target_pct"]:
            target = snap.price * (1 + s["target_pct"] / 100)
        out.append({
            "symbol": sym, "sleeve": sleeve, "action": action, "conviction": conviction, "price": snap.price,
            "horizon": str(p.get("horizon", "months" if sleeve == "long" else "days"))[:40],
            "entry_note": str(p.get("entry_note", ""))[:200], "stop": stop, "target": target,
            "reasons": clean_list(p.get("reasons"), 5), "risks": clean_list(p.get("risks"), 3),
            "news": snap.news[:3], "indicators": indicators_dict(snap),
        })
        if len(out) >= max_picks:
            break
    avoid = []
    for a in data.get("avoid", []) if isinstance(data.get("avoid"), list) else []:
        sym = str(a.get("symbol", "")).upper().strip()
        if sym in by_sym:
            avoid.append({"symbol": sym, "sleeve": sleeve, "why": str(a.get("why", ""))[:200]})
    return out, avoid


def ai_review(ai: AI, holdings: list[dict], snaps: dict[str, Snapshot], flashes_raw: list[dict], context: dict) -> dict:
    prompt = {
        "task": "1) For each holding, give advice (hold|sell|trim|add) with 2-4 bullet reasons combining the news, the numbers and the position's own stop/target/age. 2) From the news items, pick the ones that could change a decision this week (up to 8) and say why each matters and whether it's positive, negative or mixed. 3) Write a 3-5 sentence market summary for today (include the sentence that this is research, not personalised advice). 4) Write a short to-do list for today (max 6 items, imperative, specific).",
        "today": context["today"], "market_context": context["market"],
        "holdings": [{**{k: h[k] for k in ("symbol", "sleeve", "qty", "entry_price", "entry_date", "pnl_pct", "days_held", "stop", "target", "advice")},
                      "rules_view": h["reasons"][:2], **compact_snapshot(snaps[h["symbol"]], 3)} for h in holdings if h["symbol"] in snaps],
        "news": [{"id": i, "symbol": n["symbol"], "at": n["at"][:10], "headline": n["headline"]} for i, n in enumerate(flashes_raw[:40])],
        "output_schema": {
            "holdings": [{"symbol": "string", "advice": "hold|sell|trim|add", "reasons": ["2-4 bullets"]}],
            "flashes": [{"id": "news id", "why_it_matters": "one or two sentences", "impact": "positive|negative|mixed"}],
            "market_summary": "3-5 sentences", "todo": ["up to 6 items"],
        },
    }
    return ai.ask(json.dumps(prompt, separators=(",", ":")))


def to_float(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# GitHub issues for alerts (emails you automatically)
# ---------------------------------------------------------------------------

def open_issue_alerts(alerts: list[dict], run_label: str):
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    high = [a for a in alerts if a.get("severity") == "high"]
    if not token or not repo or not high:
        return
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    try:
        existing = requests.get(f"{api}/repos/{repo}/issues", headers=h, params={"state": "open", "labels": "advisor-alert", "per_page": 100}, timeout=30)
        titles = {i["title"] for i in existing.json()} if existing.ok else set()
    except requests.RequestException:
        titles = set()
    for a in high:
        title = f"Alert: {a['title']}"
        if title in titles:
            continue
        body = f"**{a['title']}**\n\n{a.get('detail', '')}\n\n**Suggested action:** {a.get('action', '')}\n\n_Raised by the {run_label} run. Close this issue once you've dealt with it._"
        try:
            requests.post(f"{api}/repos/{repo}/issues", headers=h, json={"title": title, "body": body, "labels": ["advisor-alert"]}, timeout=30)
        except requests.RequestException as e:
            log(f"Couldn't open an issue for the alert: {e}")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_label(now_et: datetime) -> str:
    h = now_et.hour
    return "morning" if h < 11 else "midday" if h < 15 else "evening"


def market_context(clock: dict, spy: Snapshot | None, btc: Snapshot | None, events: list[str]) -> dict:
    ctx = {"us_market_open": bool(clock.get("is_open")), "next_open_et": str(clock.get("next_open", ""))[:16].replace("T", " ")}
    if spy and spy.ok:
        ctx["spy"] = {"price": round(spy.price, 2), "trend": spy.trend, "d1": round(spy.day_pct or 0, 1), "w1": round(spy.week_pct or 0, 1), "m1": round(spy.month_pct or 0, 1),
                      "off_high": round(spy.off_high_pct or 0, 1), "setup": spy.setup}
    if btc and btc.ok:
        ctx["btc"] = {"price": round(btc.price, 0), "trend": btc.trend, "d1": round(btc.day_pct or 0, 1), "w1": round(btc.week_pct or 0, 1), "m1": round(btc.month_pct or 0, 1), "setup": btc.setup}
    if events:
        ctx["events"] = events
    return ctx


def load_events() -> list[str]:
    p = ROOT / "events.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")][:12]


def run(api: Alpaca, cfg: dict, portfolio: list[dict], ai: AI) -> dict:
    clock = api.clock()
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    today = now_et.date().isoformat()
    label = run_label(now_et)
    notes: list[str] = []
    long_s, short_s = cfg["sleeves"]["long"], cfg["sleeves"]["short"]

    stock_syms = list(dict.fromkeys(long_s["symbols"] + [cfg["benchmark"]] + [h["symbol"] for h in portfolio if "/" not in h["symbol"]]))
    crypto_syms = list(dict.fromkeys(short_s["symbols"] + [h["symbol"] for h in portfolio if "/" in h["symbol"]]))

    start = now_et.date() - timedelta(days=420)
    end_utc = now_utc - timedelta(minutes=16)
    try:
        sbars = api.stock_bars(stock_syms, start, end_utc, cfg["data_feed"])
    except FeedNotAllowed:
        notes.append("Your Alpaca data plan doesn't allow the 'sip' feed, so IEX prices were used.")
        sbars = api.stock_bars(stock_syms, start, end_utc, "iex")
    cbars = api.crypto_bars(crypto_syms, start)

    # bring today's bar up to the latest trade
    latest = {}
    if clock.get("is_open"):
        latest.update(api.latest_stock_trades(stock_syms))
    latest.update(api.latest_crypto_trades(crypto_syms))
    for sym, px in latest.items():
        rows = sbars.get(sym) if sym in sbars else cbars.get(sym)
        if rows is None:
            continue
        if rows and rows[-1]["d"] == today:
            b = rows[-1]
            b["c"], b["h"], b["l"] = px, max(b["h"], px), min(b["l"], px)
        elif not rows or rows[-1]["d"] < today:
            rows.append({"d": today, "o": px, "h": px, "l": px, "c": px, "v": 0})

    news_start = now_utc - timedelta(days=cfg["news_days"])
    news = api.news(stock_syms + crypto_syms, news_start)

    snaps: dict[str, Snapshot] = {}
    for sym in stock_syms:
        s = snapshot(sym, "long", sbars.get(sym, []), long_s)
        s.news = news.get(sym, [])
        snaps[sym] = s
    for sym in crypto_syms:
        s = snapshot(sym, "short", cbars.get(sym, []), short_s)
        s.news = news.get(sym, [])
        snaps[sym] = s
    missing = [s for s in stock_syms + crypto_syms if not snaps[s].price]
    if missing:
        notes.append("No price data for: " + ", ".join(missing) + " (check the tickers).")

    events = load_events()
    context = {"today": today, "run": label, "market": market_context(clock, snaps.get(cfg["benchmark"]), snaps.get("BTC/USD"), events)}

    # holdings review (rules first, AI can refine advice)
    reviews = []
    for h in portfolio:
        h2 = dict(h)
        h2["max_hold_days"] = short_s["max_hold_days"] if h["sleeve"] == "short" else 0
        reviews.append(review_holding(h2, snaps.get(h["symbol"]), today))
    held = {h["symbol"] for h in portfolio}

    # candidate screening
    def candidates(sleeve: str, syms: list[str], n: int) -> list[Snapshot]:
        pool = [snaps[s] for s in syms if snaps[s].ok and s not in held]
        pool.sort(key=lambda x: (-x.score, -(x.month_pct or 0)))
        top = pool[:n]
        newsy = [x for x in pool[n:] if x.news][: max(0, n // 3)]
        return top + newsy

    long_cands = candidates("long", long_s["symbols"], 14)
    short_cands = candidates("short", short_s["symbols"], 8)

    # raw news pool for flashes: holdings + candidates + benchmark
    flash_pool = []
    seen = set()
    for sym in [h["symbol"] for h in portfolio] + [c.symbol for c in long_cands + short_cands] + [cfg["benchmark"]]:
        for n in snaps.get(sym, Snapshot(sym, "long")).news[:3]:
            key = (n["headline"][:60])
            if key in seen:
                continue
            seen.add(key)
            flash_pool.append({**n, "symbol": sym, "impact": headline_tone(n["headline"]), "why_it_matters": ""})

    ai_info = {"provider": ai.provider, "model": ai.model, "ok": False, "error": None}
    picks_long, picks_short, avoid, flashes, summary, todo = [], [], [], [], "", []
    if ai.provider:
        try:
            picks_long, avoid_l = ai_picks(ai, "long", long_s, long_cands, context, cfg["max_long_picks"])
            picks_short, avoid_s = ai_picks(ai, "short", short_s, short_cands, context, cfg["max_short_picks"])
            avoid = avoid_l + avoid_s
            rev = ai_review(ai, reviews, snaps, flash_pool, context)
            by_sym = {r["symbol"]: r for r in reviews}
            for item in rev.get("holdings", []) if isinstance(rev.get("holdings"), list) else []:
                r = by_sym.get(str(item.get("symbol", "")).upper())
                if not r:
                    continue
                adv = item.get("advice")
                if adv in ("hold", "sell", "trim", "add"):
                    ai_reasons = clean_list(item.get("reasons"), 3)
                    if r["advice"] in ("sell", "trim", "check"):
                        # a stop, target or time limit has fired: the rules keep the final word
                        r["reasons"] = r["reasons"][:1] + ["AI view: " + x for x in ai_reasons]
                    else:
                        r["advice"] = adv
                        r["reasons"] = ai_reasons or r["reasons"]
            for f in rev.get("flashes", []) if isinstance(rev.get("flashes"), list) else []:
                try:
                    src = flash_pool[int(f.get("id"))]
                except (TypeError, ValueError, IndexError):
                    continue
                impact = f.get("impact") if f.get("impact") in ("positive", "negative", "mixed") else src["impact"]
                flashes.append({**src, "impact": impact, "why_it_matters": str(f.get("why_it_matters", ""))[:300]})
            summary = str(rev.get("market_summary", ""))[:1200]
            todo = clean_list(rev.get("todo"), 6, 160)
            ai_info["ok"] = True
        except AdvisorError as e:
            ai_info["error"] = str(e)
            notes.append(f"AI brief unavailable this run ({e}). Showing the rules-only view.")
    else:
        ai_info["error"] = "No AI configured."
        notes.append("No AI configured, so this is the rules-only view. Add ai_provider settings or a GitHub token with models: read.")

    if not picks_long:
        picks_long = [rules_pick(c, long_s, "long") for c in long_cands[: cfg["max_long_picks"]]]
    if not picks_short:
        picks_short = [rules_pick(c, short_s, "short") for c in short_cands[: cfg["max_short_picks"]]]
    if not flashes:
        flashes = [f for f in flash_pool if f["impact"] != "neutral"][:8] or flash_pool[:5]
    if not summary:
        spy = snaps.get(cfg["benchmark"])
        summary = (f"Rules-only view. {cfg['benchmark']} is in a{'n up' if spy and spy.trend == 'up' else ' down'}trend ({spy.setup if spy else 'no data'}). "
                   "Picks are ranked on trend, momentum and pullback depth; the news has not been weighed. This is research, not personalised advice.")
    if not todo:
        todo = [f"Review {len([r for r in reviews if r['advice'] != 'hold'])} holding alert(s)." if reviews else "Add your holdings to portfolio.json so they can be monitored."]
        if picks_long:
            todo.append(f"Consider the top long-term idea: {picks_long[0]['symbol']}.")

    alerts = []
    for r in reviews:
        alerts.extend(r["alerts"])
    strong = [p for p in picks_long + picks_short if p["action"] == "buy" and p["conviction"] >= 4]
    if strong:
        alerts.append({"severity": "medium", "symbol": ", ".join(p["symbol"] for p in strong),
                       "title": f"{len(strong)} strong idea{'s' if len(strong) != 1 else ''} today: " + ", ".join(p["symbol"] for p in strong),
                       "detail": "See the ideas below for the reasons, stops and how to enter.", "action": "Only act on what you understand and can size sensibly."})
    spy = snaps.get(cfg["benchmark"])
    if spy and spy.day_pct is not None and spy.day_pct <= -2:
        alerts.append({"severity": "high", "symbol": cfg["benchmark"], "title": f"Market down {spy.day_pct:.1f}% today", "detail": "A broad sell-off. Check stops on every holding.", "action": "Don't add new positions today; review stops."})
    for i, a in enumerate(alerts):
        a["id"] = f"{today}-{label}-{i}"
        a["created_at"] = now_utc.strftime("%Y-%m-%d %H:%M UTC")

    report = {
        "generated_at": now_utc.strftime("%Y-%m-%d %H:%M UTC"), "generated_at_et": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "date_et": today, "run": label, "market": {**context["market"], "summary": summary},
        "ai": ai_info, "notes": notes, "todo": todo,
        "picks": {"long": picks_long, "short": picks_short}, "avoid": avoid,
        "holdings": reviews, "alerts": alerts, "flashes": flashes,
        "universe": {"long": [compact_row(snaps[s]) for s in long_s["symbols"]], "short": [compact_row(snaps[s]) for s in short_s["symbols"]]},
        "settings": {"long": {k: long_s[k] for k in ("stop_pct", "target_pct")}, "short": {k: short_s[k] for k in ("stop_pct", "target_pct", "max_hold_days")}},
    }
    return report


def compact_row(snap: Snapshot) -> dict:
    return {"symbol": snap.symbol, "price": round(snap.price, 2) if snap.price else None, "trend": snap.trend,
            "rsi": round(snap.rsi) if snap.rsi is not None else None, "d1": round(snap.day_pct, 1) if snap.day_pct is not None else None,
            "m1": round(snap.month_pct, 1) if snap.month_pct is not None else None, "setup": snap.setup or snap.note, "news": len(snap.news)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(report: dict):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "history").mkdir(exist_ok=True)
    (DATA / "latest.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    name = f"{report['date_et']}-{report['run']}.json"
    (DATA / "history" / name).write_text(json.dumps(report, indent=1), encoding="utf-8")
    files = sorted(p.name for p in (DATA / "history").glob("*.json"))
    keep = files[-120:]
    for old in files[:-120]:
        (DATA / "history" / old).unlink(missing_ok=True)
    (DATA / "index.json").write_text(json.dumps({"latest": name, "reports": keep[::-1]}, indent=1), encoding="utf-8")
    (DOCS / ".nojekyll").touch()
    (DOCS / "report.md").write_text(report_markdown(report), encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report_markdown(report) + "\n")


def money(v):
    return f"${v:,.2f}" if isinstance(v, (int, float)) else "-"


def report_markdown(r: dict) -> str:
    L = [f"# Market brief: {r['generated_at_et']} ({r['run']} run)", "", r["market"].get("summary", ""), ""]
    if r["alerts"]:
        L += ["## Alerts", ""] + [f"- **{a['severity'].upper()}** {a['title']}: {a['detail']} _{a.get('action', '')}_" for a in r["alerts"]] + [""]
    if r["todo"]:
        L += ["## To do today", ""] + [f"- [ ] {t}" for t in r["todo"]] + [""]
    for key, title in (("long", "Long-term stock ideas"), ("short", "Short-term crypto ideas")):
        L += [f"## {title}", ""]
        for p in r["picks"][key]:
            L += [f"### {p['symbol']} — {p['action'].upper()} (conviction {p['conviction']}/5), {money(p['price'])}",
                  f"Stop {money(p['stop'])}" + (f", target {money(p['target'])}" if p.get('target') else "") + f". {p.get('entry_note', '')}", ""]
            L += [f"- {x}" for x in p["reasons"]]
            if p.get("risks"):
                L += [f"- Risk: {x}" for x in p["risks"]]
            L.append("")
    if r["holdings"]:
        L += ["## Your holdings", "", "| Symbol | Advice | Price | P&L | Stop | Why |", "|---|---|---:|---:|---:|---|"]
        for h in r["holdings"]:
            pnl = f"{h['pnl_pct']:+.1f}%" if h.get("pnl_pct") is not None else "-"
            L.append(f"| {h['symbol']} | {h['advice']} | {money(h.get('price'))} | {pnl} | {money(h.get('stop'))} | {' '.join(h['reasons'][:2])} |")
        L.append("")
    if r["flashes"]:
        L += ["## News flashes", ""] + [f"- **{f['symbol']}** ({f['impact']}): [{f['headline']}]({f['url']}) {f.get('why_it_matters', '')}" for f in r["flashes"]] + [""]
    if r["notes"]:
        L += ["## Notes", ""] + [f"- {n}" for n in r["notes"]] + [""]
    L.append("_Research for your own decisions, not personalised financial advice. Past performance doesn't predict future results._")
    return "\n".join(L)


def main() -> int:
    try:
        cfg = load_config(ROOT / "config.toml")
        portfolio = load_portfolio(ROOT / "portfolio.json", cfg)
        key = os.environ.get("ALPACA_API_KEY_ID", "").strip()
        secret = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
        if not key or not secret:
            raise AdvisorError("Missing Alpaca keys. Add ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY as GitHub repository secrets (paper keys are fine; they're only used for prices and news).")
        api = Alpaca(key, secret, os.environ.get("ALPACA_TRADING_URL", TRADING_URL), os.environ.get("ALPACA_DATA_URL", DATA_URL))
        ai = AI(cfg)
        report = run(api, cfg, portfolio, ai)
        write_outputs(report)
        if not os.environ.get("ADVISOR_NO_ISSUES"):
            open_issue_alerts(report["alerts"], report["run"])
        log(f"Brief written: {len(report['picks']['long'])} long ideas, {len(report['picks']['short'])} short ideas, {len(report['alerts'])} alerts, AI ok={report['ai']['ok']}")
        return 0
    except AdvisorError as e:
        msg = f"# Market advisor stopped\n\n**Problem:** {e}\n"
        log(msg)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as f:
                f.write(msg)
        return 1


if __name__ == "__main__":
    sys.exit(main())
