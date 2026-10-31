"""Explicit local CSV contract and a deterministic synthetic fixture; no network."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predicting_volatility_temporal import _close


@dataclass
class PriceInput:
    close: pd.Series
    available_at: pd.Series
    metadata: dict
    fingerprint: str


def utc_timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamps must include an explicit UTC offset or Z")
    return stamp.tz_convert("UTC")


def validate_prices(frame, metadata, fingerprint):
    required = {"Date", "Close", "AvailableAt"}
    if set(frame.columns) != required:
        raise ValueError("CSV must contain exactly Date, Close, AvailableAt for one asset")
    required_meta = {"schema_version", "symbol", "data_kind", "source_url", "retrieved_at", "currency", "price_adjustment", "availability_evidence", "license", "redistribution"}
    if not required_meta.issubset(metadata) or metadata["schema_version"] != 1:
        raise ValueError("Metadata does not satisfy the version 1 local data contract")
    if metadata["data_kind"] not in ("observed", "synthetic"):
        raise ValueError("data_kind must be observed or synthetic")
    if metadata["price_adjustment"] not in ("adjusted", "unadjusted", "unknown", "synthetic"):
        raise ValueError("Declare adjusted, unadjusted, unknown or synthetic prices")
    for key in required_meta - {"schema_version", "retrieved_at"}:
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"Metadata field {key} must be a nonempty string")
    if metadata["retrieved_at"] is not None:
        utc_timestamp(metadata["retrieved_at"])
    index = pd.DatetimeIndex([utc_timestamp(value) for value in frame["Date"]], name="Date")
    available = pd.Series([utc_timestamp(value) for value in frame["AvailableAt"]], index=index, name="available_at")
    close = pd.Series(pd.to_numeric(frame["Close"], errors="raise").to_numpy(), index=index, name="Close")
    if np.isinf(close.to_numpy(dtype=float)).any():
        raise ValueError("Infinite prices are invalid; missing prices must be explicit NaN")
    close = _close(close)
    if (available > index).any():
        raise ValueError("A price is unavailable at its declared forecast timestamp")
    if not available.is_monotonic_increasing:
        raise ValueError("Price availability timestamps must be nondecreasing")
    if not 100 <= len(close) <= 100_000:
        raise ValueError("This bounded project accepts between 100 and 100,000 observations")
    return PriceInput(close, available, dict(metadata), fingerprint)


def load_csv(csv_path, metadata_path):
    """Read CSV and provenance JSON exactly as supplied; never fetch or unpickle."""
    raw = Path(csv_path).read_bytes()
    metadata_raw = Path(metadata_path).read_bytes()
    metadata = json.loads(metadata_raw)
    frame = pd.read_csv(csv_path)
    fingerprint = hashlib.sha256(raw + b"\0" + metadata_raw).hexdigest()
    return validate_prices(frame, metadata, fingerprint)


def synthetic_prices(n=900, seed=42):
    """Illustrative variance regimes; not observed prices, a backtest or a claim.

    Fixed generator parameters are not calibrated to any instrument or portfolio.
    Date labels are artificial weekdays and are not an exchange holiday calendar.
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 100 <= n <= 100_000:
        raise ValueError("n must be between 100 and 100,000")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03 16:05:00", periods=n, tz="UTC", name="Date")
    regime = np.where((np.arange(n) // 90) % 3 == 1, 0.022, 0.008)
    log_returns = 0.0001 + regime * rng.standard_normal(n)
    frame = pd.DataFrame({"Date": dates, "Close": 100 * np.exp(np.cumsum(log_returns)), "AvailableAt": dates - pd.Timedelta(minutes=5)})
    metadata = {"schema_version": 1, "symbol": "SYNTHETIC", "data_kind": "synthetic", "source_url": "generated:predicting_volatility_data.synthetic_prices/v1", "retrieved_at": None, "currency": "SYNTHETIC_UNITS", "price_adjustment": "synthetic", "availability_evidence": "Constructed fixture: each artificial price is available five minutes before its forecast origin.", "license": "Generated locally; see project licensing notice.", "redistribution": "No third-party market dataset is used.", "generator": "v1", "n": n, "seed": seed}
    fingerprint = hashlib.sha256(frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode() + json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return validate_prices(frame, metadata, fingerprint)
