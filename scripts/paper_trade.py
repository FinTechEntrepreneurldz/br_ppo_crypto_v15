#!/usr/bin/env python3
"""
br_ppo_crypto_v15 paper-trading runner.

This runner is intentionally fail-closed for PPO mode:
  - ALLOCATION_MODE=ppo uses the shipped PPO/BR-PPO model artifacts.
  - REQUIRE_PPO=true means the script will NOT silently trade DEFAULT_ACTION if PPO inference fails.
  - DEFAULT_ACTION is only used when ALLOCATION_MODE=fixed, or if REQUIRE_PPO=false.

GitHub Actions secrets expected:
  ALPACA_CRYPTO_V15_KEY_ID
  ALPACA_CRYPTO_V15_SECRET_KEY

Recommended GitHub Actions variables:
  ALLOCATION_MODE=ppo
  REQUIRE_PPO=true
  INCLUDE_PRIMARY_MODEL=false
  SUBMIT_ORDERS=false
  CANCEL_OPEN_ORDERS=true
  FORCE_REBALANCE=false
  REBALANCE_EVERY_DAYS=10
  MIN_ORDER_NOTIONAL=25
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

METADATA_PATH = MODELS_DIR / "v8_crypto_mandatory_freqai_sharpe_max_v84_metadata.json"
PRIMARY_MODEL_PATH = MODELS_DIR / "v8_crypto_mandatory_freqai_sharpe_max_v84_primary.zip"
ENSEMBLE_MODEL_PATHS = [
    MODELS_DIR / "v8_crypto_mandatory_freqai_sharpe_max_v84_member_0.zip",
    MODELS_DIR / "v8_crypto_mandatory_freqai_sharpe_max_v84_member_1.zip",
    MODELS_DIR / "v8_crypto_mandatory_freqai_sharpe_max_v84_member_2.zip",
]

PORTFOLIO_CSV = LOGS_DIR / "portfolio" / "portfolio.csv"
LATEST_DECISION_CSV = LOGS_DIR / "decisions" / "latest_decision.csv"
DECISIONS_CSV = LOGS_DIR / "decisions" / "decisions.csv"
LATEST_TARGET_WEIGHTS_CSV = LOGS_DIR / "target_weights" / "latest_target_weights.csv"
TARGET_WEIGHTS_CSV = LOGS_DIR / "target_weights" / "target_weights.csv"
LATEST_POSITIONS_CSV = LOGS_DIR / "positions" / "latest_positions.csv"
LATEST_PLANNED_ORDERS_CSV = LOGS_DIR / "orders" / "latest_planned_orders.csv"
LATEST_SUBMITTED_ORDERS_CSV = LOGS_DIR / "orders" / "latest_submitted_orders.csv"
SUBMITTED_ORDERS_CSV = LOGS_DIR / "orders" / "submitted_orders.csv"
HEALTH_STATUS_JSON = LOGS_DIR / "health" / "health_status.json"
SIGNAL_HISTORY_CSV = LOGS_DIR / "health" / "signal_history.csv"

MODEL_ID = "br_ppo_crypto_v15"


@dataclass
class Settings:
    allocation_mode: str
    default_action: str
    require_ppo: bool
    include_primary_model: bool
    submit_orders: bool
    cancel_open_orders: bool
    min_order_notional: float
    rebalance_every_days: int
    force_rebalance: bool
    alpaca_key_id: str
    alpaca_secret_key: str
    alpaca_base_url: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_str() -> str:
    return utc_now().isoformat()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def ensure_dirs() -> None:
    for path in [
        PORTFOLIO_CSV,
        LATEST_DECISION_CSV,
        DECISIONS_CSV,
        LATEST_TARGET_WEIGHTS_CSV,
        TARGET_WEIGHTS_CSV,
        LATEST_POSITIONS_CSV,
        LATEST_PLANNED_ORDERS_CSV,
        LATEST_SUBMITTED_ORDERS_CSV,
        SUBMITTED_ORDERS_CSV,
        HEALTH_STATUS_JSON,
        SIGNAL_HISTORY_CSV,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    return Settings(
        allocation_mode=os.getenv("ALLOCATION_MODE", "ppo").strip().lower(),
        default_action=os.getenv("DEFAULT_ACTION", "crypto10_freqai15_llm25_ichi25_bil25").strip(),
        require_ppo=env_bool("REQUIRE_PPO", True),
        include_primary_model=env_bool("INCLUDE_PRIMARY_MODEL", False),
        submit_orders=env_bool("SUBMIT_ORDERS", False),
        cancel_open_orders=env_bool("CANCEL_OPEN_ORDERS", False),
        min_order_notional=env_float("MIN_ORDER_NOTIONAL", 25.0),
        rebalance_every_days=env_int("REBALANCE_EVERY_DAYS", 10),
        force_rebalance=env_bool("FORCE_REBALANCE", False),
        alpaca_key_id=os.getenv("ALPACA_CRYPTO_V15_KEY_ID", "").strip(),
        alpaca_secret_key=os.getenv("ALPACA_CRYPTO_V15_SECRET_KEY", "").strip(),
        alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/"),
    )


def load_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")
    return json.loads(METADATA_PATH.read_text())


def model_paths(settings: Settings) -> List[Path]:
    paths = [p for p in ENSEMBLE_MODEL_PATHS if p.exists()]
    if settings.include_primary_model and PRIMARY_MODEL_PATH.exists():
        paths = [PRIMARY_MODEL_PATH] + paths
    if not paths and PRIMARY_MODEL_PATH.exists():
        paths = [PRIMARY_MODEL_PATH]
    if not paths:
        missing = [str(p) for p in [PRIMARY_MODEL_PATH] + ENSEMBLE_MODEL_PATHS]
        raise FileNotFoundError(f"No PPO artifacts found. Expected one of: {missing}")
    return paths


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {str(k): float(v) for k, v in weights.items() if np.isfinite(float(v)) and abs(float(v)) > 1e-12}
    total = sum(clean.values())
    if abs(total) <= 1e-12:
        return {}
    return {k: v / total for k, v in clean.items()}


# -------------------------
# Market/proxy feature layer
# -------------------------

def _extract_close(downloaded: pd.DataFrame) -> pd.DataFrame:
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" in downloaded.columns.get_level_values(-1):
            close = downloaded.xs("Close", axis=1, level=-1)
        elif "Close" in downloaded.columns.get_level_values(0):
            close = downloaded.xs("Close", axis=1, level=0)
        else:
            return pd.DataFrame()
        close.columns = [str(c) for c in close.columns]
        return close
    if "Close" in downloaded.columns:
        return downloaded[["Close"]]
    return downloaded


def fetch_close_prices(period: str = "240d") -> pd.DataFrame:
    tickers = [
        "SPY", "QQQ", "VTI", "RSP", "IWM", "BIL",
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "AVAX-USD", "DOGE-USD", "LINK-USD", "LTC-USD",
    ]
    try:
        import yfinance as yf
        raw = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        close = _extract_close(raw).sort_index()
        close = close.loc[:, ~close.columns.duplicated()]
        return close.dropna(axis=1, how="all")
    except Exception:
        return pd.DataFrame()


def pct_change_series(close: pd.Series) -> pd.Series:
    s = pd.Series(close).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return pd.Series(dtype=float)
    return s.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()


def weighted_stream(returns: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    frames = []
    clean_weights = {}
    for symbol, weight in weights.items():
        s = returns.get(symbol)
        if s is not None and len(s):
            frames.append(s.rename(symbol))
            clean_weights[symbol] = float(weight)
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames, axis=1).fillna(0.0)
    w = pd.Series(clean_weights).reindex(df.columns).fillna(0.0)
    if abs(w.sum()) > 1e-12:
        w = w / w.sum()
    return df.dot(w)


def top_momentum_crypto_stream(returns: Dict[str, pd.Series]) -> pd.Series:
    crypto = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "LINK-USD", "LTC-USD"]
    scores = {}
    for sym in crypto:
        r = returns.get(sym)
        if r is not None and len(r) >= 21:
            scores[sym] = float((1.0 + r.tail(63).fillna(0.0)).prod() - 1.0)
    if not scores:
        return weighted_stream(returns, {"BTC-USD": 0.5, "ETH-USD": 0.5})
    top = sorted(scores, key=scores.get, reverse=True)[:3]
    return weighted_stream(returns, {s: 1.0 / len(top) for s in top})


def low_vol_crypto_stream(returns: Dict[str, pd.Series]) -> pd.Series:
    crypto = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "LINK-USD", "LTC-USD"]
    vols = {}
    for sym in crypto:
        r = returns.get(sym)
        if r is not None and len(r) >= 21:
            vols[sym] = float(r.tail(63).std())
    if not vols:
        return weighted_stream(returns, {"BTC-USD": 0.5, "ETH-USD": 0.5})
    picks = sorted(vols, key=vols.get)[:3]
    return weighted_stream(returns, {s: 1.0 / len(picks) for s in picks})


def crypto_trend_gate_from_returns(returns: Dict[str, pd.Series]) -> float:
    def trending(sym: str) -> bool:
        r = returns.get(sym)
        if r is None or len(r) < 126:
            return False
        eq = (1.0 + r.fillna(0.0)).cumprod()
        if len(eq) < 126:
            return False
        above_ma = eq.iloc[-1] > eq.rolling(126).mean().iloc[-1]
        positive_63 = eq.iloc[-1] / max(eq.iloc[-63], 1e-12) - 1.0 > 0
        return bool(above_ma and positive_63)
    return 1.0 if trending("BTC-USD") or trending("ETH-USD") else 0.0


def build_proxy_stream_returns(close: pd.DataFrame) -> Dict[str, pd.Series]:
    returns: Dict[str, pd.Series] = {}
    if close is None or close.empty:
        return returns

    base_returns = {c: pct_change_series(close[c]) for c in close.columns}
    for k, v in base_returns.items():
        returns[k] = v

    returns["TOP_EW"] = weighted_stream(base_returns, {"SPY": 0.55, "QQQ": 0.25, "RSP": 0.20})
    returns["CURRENT_EW"] = weighted_stream(base_returns, {"SPY": 0.60, "QQQ": 0.25, "VTI": 0.15})

    crypto_core = weighted_stream(base_returns, {"BTC-USD": 0.50, "ETH-USD": 0.50})
    crypto_top = top_momentum_crypto_stream(base_returns)
    crypto_low_vol = low_vol_crypto_stream(base_returns)
    gate = crypto_trend_gate_from_returns(base_returns)
    bil = base_returns.get("BIL", pd.Series(0.0, index=crypto_core.index if len(crypto_core) else None))

    crypto_like = {
        "CRYPTO_BTC_ETH_CORE": crypto_core,
        "CRYPTO_LONG_MOM": crypto_top,
        "CRYPTO_ICHIMOKU_TREND": crypto_top,
        "CRYPTO_LS": crypto_top,
        "CRYPTO_RISK_PARITY": weighted_stream(base_returns, {"BTC-USD": 0.35, "ETH-USD": 0.35, "SOL-USD": 0.20, "BNB-USD": 0.10}),
        "CRYPTO_BREAKOUT": crypto_top,
        "CRYPTO_REVERSAL": crypto_low_vol,
        "CRYPTO_ALT_MOM": weighted_stream(base_returns, {"SOL-USD": 0.35, "BNB-USD": 0.25, "XRP-USD": 0.20, "AVAX-USD": 0.20}),
        "CRYPTO_LOW_VOL_CORE": crypto_low_vol,
        "CRYPTO_VOL_TARGETED_TREND": crypto_top,
        "CRYPTO_FREQAI_ML_LONG": crypto_top,
        "CRYPTO_FREQAI_ML_LS": crypto_top,
        "CRYPTO_FREQAI_CONFIDENCE": crypto_top,
        "CRYPTO_FREQAI_META_LONG": crypto_top,
        "CRYPTO_FREQAI_META_LS": crypto_top,
        "CRYPTO_FREQAI_META_CONFIDENCE": crypto_top,
    }
    for k, v in crypto_like.items():
        returns[k] = v

    gated_streams = ["CRYPTO_CASH_GATED", "CRYPTO_FREQAI_CASH_GATED", "CRYPTO_FREQAI_META_CASH_GATED", "CRYPTO_CRASH_DEFENSIVE"]
    for k in gated_streams:
        if len(crypto_top):
            aligned = pd.concat([crypto_top.rename("crypto"), bil.rename("bil")], axis=1).fillna(0.0)
            returns[k] = aligned["crypto"] * gate + aligned["bil"] * (1.0 - gate)
        else:
            returns[k] = bil

    # Equity/alpha proxy streams. These are not exact research sleeves, but give live market state to the PPO artifact.
    returns["V8_ALPHA_ENSEMBLE"] = weighted_stream(base_returns, {"SPY": 0.55, "QQQ": 0.35, "BIL": 0.10})
    returns["V8_ALPHA_MOMENTUM_LONG"] = weighted_stream(base_returns, {"SPY": 0.60, "QQQ": 0.40})
    returns["V8_ALPHA_ICHIMOKU_LONG"] = weighted_stream(base_returns, {"SPY": 0.50, "QQQ": 0.40, "BIL": 0.10})
    returns["V8_ALPHA_LLM_LONG"] = weighted_stream(base_returns, {"SPY": 0.45, "QQQ": 0.45, "BIL": 0.10})
    returns["V8_ALPHA_LS"] = weighted_stream(base_returns, {"SPY": 0.40, "QQQ": 0.35, "BIL": 0.25})
    returns["V8_ALPHA_LLM_LS"] = weighted_stream(base_returns, {"SPY": 0.35, "QQQ": 0.35, "BIL": 0.30})

    # Allocator proxies.
    returns["V82_HRP_LITE_ALLOCATOR"] = weighted_stream(base_returns, {"SPY": 0.30, "QQQ": 0.20, "BIL": 0.40, "BTC-USD": 0.05, "ETH-USD": 0.05})
    returns["V82_SIMPLE_META_ALLOCATOR"] = weighted_stream(base_returns, {"SPY": 0.35, "QQQ": 0.25, "BIL": 0.30, "BTC-USD": 0.05, "ETH-USD": 0.05})
    returns["V84_NETWORK_RP_ALLOCATOR"] = weighted_stream(base_returns, {"SPY": 0.30, "QQQ": 0.25, "BIL": 0.30, "BTC-USD": 0.075, "ETH-USD": 0.075})

    return returns


def compound_return(r: pd.Series) -> float:
    x = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return 0.0
    return float((1.0 + x).prod() - 1.0)


def annualized_vol(r: pd.Series) -> float:
    x = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return 0.0
    return float(x.std() * np.sqrt(252))


def annualized_sharpe(r: pd.Series) -> float:
    x = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2 or x.std() <= 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(252))


def max_drawdown(r: pd.Series) -> float:
    x = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return 0.0
    eq = (1.0 + x).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def feature_value(feature_name: str, streams: Dict[str, pd.Series]) -> float:
    # expected examples:
    # CRYPTO_BTC_ETH_CORE_ret_10
    # CRYPTO_LONG_MOM_minus_BIL_sharpe_63
    parts = feature_name.rsplit("_", 2)
    if len(parts) != 3:
        return 0.0
    stream_expr, metric, window_raw = parts
    if metric not in {"ret", "vol", "sharpe", "dd"}:
        return 0.0
    try:
        window = int(window_raw)
    except Exception:
        return 0.0

    if "_minus_" in stream_expr:
        left, right = stream_expr.split("_minus_", 1)
        a = streams.get(left, pd.Series(dtype=float))
        b = streams.get(right, pd.Series(dtype=float))
        s = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0.0)
        r = s["a"] - s["b"]
    else:
        r = streams.get(stream_expr, pd.Series(dtype=float))

    r = pd.Series(r).tail(window).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return 0.0
    if metric == "ret":
        value = compound_return(r)
    elif metric == "vol":
        value = annualized_vol(r)
    elif metric == "sharpe":
        value = annualized_sharpe(r)
    else:
        value = max_drawdown(r)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -10.0, 10.0))


def build_live_feature_vector(metadata: Dict[str, Any]) -> np.ndarray:
    feature_cols = metadata.get("feature_cols") or []
    close = fetch_close_prices()
    streams = build_proxy_stream_returns(close)
    values = [feature_value(str(name), streams) for name in feature_cols]
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(arr, -10.0, 10.0)


def default_last_action(metadata: Dict[str, Any], action_names: List[str]) -> str:
    candidates = [
        metadata.get("latest_action"),
        os.getenv("LAST_ACTION", "").strip() or None,
        os.getenv("DEFAULT_ACTION", "").strip() or None,
        "crypto10_freqai15_llm25_ichi25_bil25",
        "crypto05_bil95",
    ]
    for name in candidates:
        if name in action_names:
            return str(name)
    return action_names[0]


def build_observation_for_model(model: Any, metadata: Dict[str, Any], live_features: np.ndarray) -> Tuple[np.ndarray, str]:
    expected_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if not expected_shape or len(expected_shape) != 1:
        raise ValueError(f"Unsupported PPO observation shape: {expected_shape}")

    obs_dim = int(expected_shape[0])
    obs = np.zeros(obs_dim, dtype=np.float32)

    action_names = list(metadata.get("action_names") or [])
    onehot_len = len(action_names)
    feature_slots = max(0, obs_dim - onehot_len)

    n = min(feature_slots, len(live_features))
    if n:
        obs[:n] = live_features[:n]

    if onehot_len and obs_dim >= onehot_len:
        onehot_start = obs_dim - onehot_len
        last_action = default_last_action(metadata, action_names)
        obs[onehot_start + action_names.index(last_action)] = 1.0

    return obs, f"live_proxy_features_exact_shape_{obs_dim}"


def action_probabilities(model: Any, obs: np.ndarray, n_actions: int) -> np.ndarray:
    try:
        import torch
        obs_t = torch.as_tensor(obs).float().unsqueeze(0).to(model.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
        probs = np.asarray(probs, dtype=float)
        if probs.ndim == 1 and len(probs) == n_actions and np.all(np.isfinite(probs)) and probs.sum() > 0:
            return probs / probs.sum()
    except Exception:
        pass

    action_idx, _ = model.predict(obs, deterministic=True)
    idx = int(action_idx)
    if idx < 0 or idx >= n_actions:
        raise ValueError(f"PPO returned invalid action index {idx}")
    probs = np.zeros(n_actions, dtype=float)
    probs[idx] = 1.0
    return probs


def choose_action_with_ppo(metadata: Dict[str, Any], settings: Settings) -> Tuple[str, str, str, bool, str, Dict[str, Any]]:
    from stable_baselines3 import PPO

    action_names = list(metadata.get("action_names") or [])
    if not action_names:
        raise ValueError("metadata.action_names missing")

    paths = model_paths(settings)
    live_features = build_live_feature_vector(metadata)

    prob_rows = []
    allocation_state = "ppo_unset"
    model_errors = []

    for path in paths:
        try:
            model = PPO.load(str(path))
            obs, allocation_state = build_observation_for_model(model, metadata, live_features)
            probs = action_probabilities(model, obs, len(action_names))
            prob_rows.append(probs)
        except Exception as exc:
            model_errors.append({"model_path": str(path), "error": repr(exc)})

    if not prob_rows:
        raise RuntimeError(f"All PPO models failed inference: {model_errors}")

    avg_probs = np.nanmean(np.vstack(prob_rows), axis=0)
    idx = int(np.nanargmax(avg_probs))
    if idx < 0 or idx >= len(action_names):
        raise ValueError(f"invalid action index from PPO ensemble: {idx}")

    action = action_names[idx]
    debug = {
        "n_models_loaded": len(prob_rows),
        "n_model_errors": len(model_errors),
        "model_errors": model_errors,
        "chosen_action_index": idx,
        "chosen_action_probability": float(avg_probs[idx]),
        "top_action_probabilities": [
            {"action": action_names[i], "probability": float(avg_probs[i])}
            for i in np.argsort(avg_probs)[::-1][:5]
        ],
        "live_feature_count": int(len(live_features)),
    }
    return action, "ppo_ensemble_model", allocation_state, False, "", debug


def choose_action(settings: Settings, metadata: Dict[str, Any]) -> Tuple[str, str, str, bool, str, Dict[str, Any]]:
    if settings.allocation_mode == "ppo":
        try:
            return choose_action_with_ppo(metadata, settings)
        except Exception as exc:
            if settings.require_ppo:
                raise RuntimeError(f"REQUIRE_PPO=true and PPO inference failed: {repr(exc)}") from exc
            return (
                settings.default_action,
                "default_action_fallback",
                "fixed_default",
                True,
                repr(exc),
                {"ppo_error": repr(exc)},
            )

    return settings.default_action, "default_action", "fixed_default", False, "", {}


# -------------------------
# Target allocation mapping
# -------------------------

def add_weight(targets: Dict[str, float], symbol: str, weight: float) -> None:
    if weight is None:
        return
    w = float(weight)
    if np.isfinite(w) and abs(w) > 1e-12:
        targets[symbol] = targets.get(symbol, 0.0) + w


def live_crypto_gate() -> float:
    close = fetch_close_prices(period="180d")
    returns = {c: pct_change_series(close[c]) for c in close.columns} if close is not None and not close.empty else {}
    return crypto_trend_gate_from_returns(returns)


def map_crypto_basket(targets: Dict[str, float], weight: float, aggressive: bool = False) -> None:
    if aggressive:
        add_weight(targets, "BTC/USD", weight * 0.35)
        add_weight(targets, "ETH/USD", weight * 0.30)
        add_weight(targets, "SOL/USD", weight * 0.20)
        add_weight(targets, "LINK/USD", weight * 0.15)
    else:
        add_weight(targets, "BTC/USD", weight * 0.50)
        add_weight(targets, "ETH/USD", weight * 0.35)
        add_weight(targets, "SOL/USD", weight * 0.15)


def flatten_action_to_tradeable_targets(action_weights: Dict[str, float]) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    gate_cache: float | None = None

    for stream, weight in action_weights.items():
        w = float(weight)
        u = str(stream).upper()

        if u == "BIL":
            add_weight(targets, "BIL", w)
        elif u in {"SPY", "QQQ", "VTI", "RSP", "IWM"}:
            add_weight(targets, u, w)
        elif u in {"TOP_EW", "CURRENT_EW"}:
            add_weight(targets, "SPY", w * 0.65)
            add_weight(targets, "QQQ", w * 0.25)
            add_weight(targets, "VTI", w * 0.10)
        elif u == "CRYPTO_BTC_ETH_CORE":
            add_weight(targets, "BTC/USD", w * 0.55)
            add_weight(targets, "ETH/USD", w * 0.45)
        elif "CASH_GATED" in u or "CRASH_DEFENSIVE" in u:
            if gate_cache is None:
                gate_cache = live_crypto_gate()
            crypto_w = w * gate_cache
            cash_w = w - crypto_w
            map_crypto_basket(targets, crypto_w, aggressive=False)
            add_weight(targets, "BIL", cash_w)
        elif "CRYPTO" in u:
            map_crypto_basket(targets, w, aggressive=True)
        elif "NETWORK_RP" in u or "HRP" in u or "SIMPLE_META" in u:
            add_weight(targets, "SPY", w * 0.30)
            add_weight(targets, "QQQ", w * 0.20)
            add_weight(targets, "BIL", w * 0.35)
            map_crypto_basket(targets, w * 0.15, aggressive=False)
        elif "ALPHA" in u or "LLM" in u or "ICHIMOKU" in u or "MOMENTUM" in u:
            add_weight(targets, "SPY", w * 0.55)
            add_weight(targets, "QQQ", w * 0.35)
            add_weight(targets, "BIL", w * 0.10)
        else:
            add_weight(targets, "BIL", w)

    return normalize_weights(targets)


# -------------------------
# Alpaca helpers
# -------------------------

def alpaca_headers(settings: Settings) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": settings.alpaca_key_id,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        "Content-Type": "application/json",
    }


def alpaca_get(settings: Settings, path: str) -> Any:
    response = requests.get(f"{settings.alpaca_base_url}{path}", headers=alpaca_headers(settings), timeout=30)
    response.raise_for_status()
    return response.json()


def alpaca_post(settings: Settings, path: str, payload: Dict[str, Any]) -> Any:
    response = requests.post(f"{settings.alpaca_base_url}{path}", headers=alpaca_headers(settings), json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def alpaca_delete(settings: Settings, path: str) -> Any:
    response = requests.delete(f"{settings.alpaca_base_url}{path}", headers=alpaca_headers(settings), timeout=30)
    response.raise_for_status()
    if not response.text:
        return {}
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def cancel_open_orders(settings: Settings) -> Dict[str, Any]:
    if not settings.alpaca_key_id or not settings.alpaca_secret_key:
        return {"cancel_status": "missing_credentials", "cancelled_orders": 0}
    try:
        result = alpaca_delete(settings, "/v2/orders")
        return {"cancel_status": "ok", "cancelled_orders": len(result) if isinstance(result, list) else 0}
    except Exception as exc:
        return {"cancel_status": f"cancel_error:{repr(exc)[:220]}", "cancelled_orders": 0}


def get_account(settings: Settings) -> Dict[str, Any]:
    if not settings.alpaca_key_id or not settings.alpaca_secret_key:
        return {
            "status": "missing_credentials",
            "portfolio_value": 1_000_000.0,
            "equity": 1_000_000.0,
            "cash": 1_000_000.0,
            "buying_power": 2_000_000.0,
        }
    try:
        account = alpaca_get(settings, "/v2/account")
        return {
            "status": "connected",
            "portfolio_value": float(account.get("portfolio_value", 0.0)),
            "equity": float(account.get("equity", 0.0)),
            "cash": float(account.get("cash", 0.0)),
            "buying_power": float(account.get("buying_power", 0.0)),
        }
    except Exception as exc:
        return {
            "status": f"account_error:{repr(exc)[:180]}",
            "portfolio_value": 1_000_000.0,
            "equity": 1_000_000.0,
            "cash": 1_000_000.0,
            "buying_power": 2_000_000.0,
        }


def get_positions(settings: Settings) -> List[Dict[str, Any]]:
    if not settings.alpaca_key_id or not settings.alpaca_secret_key:
        return []
    try:
        positions = alpaca_get(settings, "/v2/positions")
        return positions if isinstance(positions, list) else []
    except Exception:
        return []


def normalize_trade_symbol(symbol: str) -> str:
    s = str(symbol).upper().replace("-USD", "USD")
    if s in {"BTCUSD", "BTC/USD"}:
        return "BTC/USD"
    if s in {"ETHUSD", "ETH/USD"}:
        return "ETH/USD"
    if s in {"SOLUSD", "SOL/USD"}:
        return "SOL/USD"
    if s in {"LINKUSD", "LINK/USD"}:
        return "LINK/USD"
    return str(symbol).upper()


def symbol_from_position(position: Dict[str, Any]) -> str:
    raw = str(position.get("symbol") or "")
    return normalize_trade_symbol(raw)


def current_position_values(positions: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in positions:
        symbol = symbol_from_position(p)
        try:
            value = float(p.get("market_value", 0.0))
        except Exception:
            value = 0.0
        out[symbol] = out.get(symbol, 0.0) + value
    return out


def build_planned_orders(
    account: Dict[str, Any],
    positions: List[Dict[str, Any]],
    target_weights: Dict[str, float],
    min_order_notional: float,
) -> List[Dict[str, Any]]:
    equity = float(account.get("portfolio_value") or account.get("equity") or 0.0)
    current = current_position_values(positions)
    symbols = sorted(set(target_weights) | set(current))
    rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        target_value = equity * float(target_weights.get(symbol, 0.0))
        current_value = float(current.get(symbol, 0.0))
        delta_value = target_value - current_value

        if abs(delta_value) < min_order_notional:
            continue

        is_crypto = "/" in symbol
        rows.append({
            "timestamp_utc": utc_now_str(),
            "symbol": symbol,
            "side": "buy" if delta_value > 0 else "sell",
            "notional": round(abs(delta_value), 2),
            "target_value": round(target_value, 2),
            "current_value": round(current_value, 2),
            "delta_value": round(delta_value, 2),
            "order_type": "market",
            "time_in_force": "gtc" if is_crypto else "day",
        })
    return rows


def submit_orders(settings: Settings, planned: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    submitted: List[Dict[str, Any]] = []
    if not settings.submit_orders:
        return submitted
    if not settings.alpaca_key_id or not settings.alpaca_secret_key:
        return submitted

    for order in planned:
        payload = {
            "symbol": normalize_trade_symbol(order["symbol"]),
            "side": order["side"],
            "type": "market",
            "time_in_force": order.get("time_in_force", "day"),
            "notional": str(order["notional"]),
        }
        try:
            result = alpaca_post(settings, "/v2/orders", payload)
            submitted.append({
                **order,
                "submitted": True,
                "alpaca_order_id": result.get("id", ""),
                "status": result.get("status", ""),
                "submitted_at": utc_now_str(),
            })
        except Exception as exc:
            submitted.append({
                **order,
                "submitted": False,
                "alpaca_order_id": "",
                "status": f"submit_error:{repr(exc)[:260]}",
                "submitted_at": utc_now_str(),
            })
        time.sleep(0.2)
    return submitted


# -------------------------
# Logging + rebalance gate
# -------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not append:
            path.write_text("")
        return
    fields = list(rows[0].keys())
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def portfolio_row(account: Dict[str, Any], positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    long_value = 0.0
    short_value = 0.0
    for p in positions:
        try:
            value = float(p.get("market_value", 0.0))
        except Exception:
            value = 0.0
        if value >= 0:
            long_value += value
        else:
            short_value += value
    return {
        "timestamp_utc": utc_now_str(),
        "portfolio_value": round(float(account.get("portfolio_value", 0.0)), 2),
        "equity": round(float(account.get("equity", 0.0)), 2),
        "cash": round(float(account.get("cash", 0.0)), 2),
        "long_value": round(long_value, 2),
        "short_value": round(short_value, 2),
        "buying_power": round(float(account.get("buying_power", 0.0)), 2),
        "n_positions": len(positions),
    }


def position_rows(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for p in positions:
        rows.append({
            "timestamp_utc": utc_now_str(),
            "symbol": symbol_from_position(p),
            "qty": p.get("qty", ""),
            "market_value": p.get("market_value", ""),
            "avg_entry_price": p.get("avg_entry_price", ""),
            "current_price": p.get("current_price", ""),
            "unrealized_pl": p.get("unrealized_pl", ""),
            "unrealized_plpc": p.get("unrealized_plpc", ""),
        })
    return rows


def target_weight_rows(targets: Dict[str, float], action: str) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp_utc": utc_now_str(),
            "action": action,
            "symbol": s,
            "target_weight": round(float(w), 8),
        }
        for s, w in sorted(targets.items())
    ]


def last_trade_decision_date() -> datetime | None:
    if not DECISIONS_CSV.exists() or DECISIONS_CSV.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(DECISIONS_CSV)
        if "orders_allowed" in df.columns:
            df = df[df["orders_allowed"].astype(str).str.lower().eq("true")]
        if df.empty:
            return None
        return pd.to_datetime(df["timestamp_utc"].iloc[-1], utc=True).to_pydatetime()
    except Exception:
        return None


def should_rebalance(settings: Settings) -> Tuple[bool, str]:
    if settings.force_rebalance:
        return True, "force_rebalance"
    if settings.rebalance_every_days <= 0:
        return True, "rebalance_gate_disabled"
    last = last_trade_decision_date()
    if last is None:
        return True, "first_run"
    elapsed = (utc_now() - last).days
    if elapsed >= settings.rebalance_every_days:
        return True, f"elapsed_days_{elapsed}"
    return False, f"rebalance_wait_elapsed_days_{elapsed}_threshold_{settings.rebalance_every_days}"


def main() -> int:
    ensure_dirs()
    settings = load_settings()
    metadata = load_metadata()
    action_specs = metadata.get("action_specs") or {}

    action, source, state, fallback, fallback_reason, ppo_debug = choose_action(settings, metadata)

    if action not in action_specs:
        raise RuntimeError(f"Selected action is not in metadata.action_specs: {action}")

    action_weights = normalize_weights(action_specs.get(action, {}))
    target_weights = flatten_action_to_tradeable_targets(action_weights)

    account = get_account(settings)
    positions = get_positions(settings)
    orders_allowed, rebalance_reason = should_rebalance(settings)

    cancel_result = {"cancel_status": "not_requested", "cancelled_orders": 0}
    if orders_allowed and settings.cancel_open_orders and settings.submit_orders:
        cancel_result = cancel_open_orders(settings)
        time.sleep(2)

    planned = build_planned_orders(account, positions, target_weights, settings.min_order_notional) if orders_allowed else []
    submitted = submit_orders(settings, planned) if orders_allowed else []

    decision = {
        "timestamp_utc": utc_now_str(),
        "model_id": MODEL_ID,
        "allocation_mode": settings.allocation_mode,
        "action": action,
        "action_source": source,
        "allocation_state": state,
        "fallback_used": str(bool(fallback)).lower(),
        "fallback_reason": fallback_reason,
        "require_ppo": str(bool(settings.require_ppo)).lower(),
        "submit_orders": str(bool(settings.submit_orders)).lower(),
        "orders_allowed": str(bool(orders_allowed)).lower(),
        "rebalance_reason": rebalance_reason,
        "cancel_open_orders": str(bool(settings.cancel_open_orders)).lower(),
        "cancel_status": cancel_result.get("cancel_status"),
        "cancelled_orders": cancel_result.get("cancelled_orders"),
        "account_status": account.get("status", "unknown"),
        "portfolio_value": account.get("portfolio_value"),
        "equity": account.get("equity"),
        "cash": account.get("cash"),
        "n_target_symbols": len(target_weights),
        "n_planned_orders": len(planned),
        "n_submitted_orders": len(submitted),
        "selected_variant": metadata.get("selected_variant", ""),
        "metadata_name": metadata.get("name", ""),
        "ppo_models_loaded": ppo_debug.get("n_models_loaded", ""),
        "ppo_chosen_probability": ppo_debug.get("chosen_action_probability", ""),
    }

    write_csv(PORTFOLIO_CSV, [portfolio_row(account, positions)], append=True)
    write_csv(LATEST_DECISION_CSV, [decision], append=False)
    write_csv(DECISIONS_CSV, [decision], append=True)
    write_csv(LATEST_TARGET_WEIGHTS_CSV, target_weight_rows(target_weights, action), append=False)
    write_csv(TARGET_WEIGHTS_CSV, target_weight_rows(target_weights, action), append=True)
    write_csv(LATEST_POSITIONS_CSV, position_rows(positions), append=False)
    write_csv(LATEST_PLANNED_ORDERS_CSV, planned, append=False)
    write_csv(LATEST_SUBMITTED_ORDERS_CSV, submitted, append=False)
    write_csv(SUBMITTED_ORDERS_CSV, submitted, append=True)

    health = {
        "timestamp_utc": utc_now_str(),
        "model_id": MODEL_ID,
        "overall_status": "ok" if not fallback else "fallback",
        "allocation_mode": settings.allocation_mode,
        "action": action,
        "action_source": source,
        "allocation_state": state,
        "fallback_used": fallback,
        "fallback_reason": fallback_reason,
        "require_ppo": settings.require_ppo,
        "orders_allowed": orders_allowed,
        "rebalance_reason": rebalance_reason,
        "submit_orders": settings.submit_orders,
        "cancel_open_orders": settings.cancel_open_orders,
        "cancel_status": cancel_result.get("cancel_status"),
        "cancelled_orders": cancel_result.get("cancelled_orders"),
        "account_status": account.get("status", "unknown"),
        "target_weights": target_weights,
        "action_weights": action_weights,
        "ppo_debug": ppo_debug,
    }
    write_json(HEALTH_STATUS_JSON, health)

    write_csv(
        SIGNAL_HISTORY_CSV,
        [{
            "timestamp_utc": utc_now_str(),
            "model_id": MODEL_ID,
            "signal": action,
            "allocation_mode": settings.allocation_mode,
            "action_source": source,
            "fallback_used": str(bool(fallback)).lower(),
            "fallback_reason": fallback_reason,
        }],
        append=True,
    )

    print(json.dumps(decision, indent=2, default=str))
    print("Target weights:")
    print(json.dumps(target_weights, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
