import warnings
import os
from dotenv import load_dotenv

load_dotenv()

# Suppress the joblib/scikit-learn parallel delayed UserWarning emitted by LightGBM/XGBoost/sklearn
# This is an internal library warning that does not affect correctness.
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

def _parse_budget_map(raw: str, default: dict[str, int]) -> dict[str, int]:
    text = (raw or "").strip()
    if not text:
        return dict(default)
    parsed: dict[str, int] = {}
    for part in text.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().upper()
        try:
            parsed[key] = int(value.strip())
        except Exception:
            continue
    return parsed or dict(default)

def _parse_csv_tuple(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    text = (raw or "").strip()
    if not text:
        return tuple(default)
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    return values or tuple(default)

def _parse_float_map(raw: str, default: dict[str, float]) -> dict[str, float]:
    text = (raw or "").strip()
    if not text:
        return dict(default)
    parsed: dict[str, float] = {}
    for part in text.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        try:
            parsed[key] = float(value.strip())
        except Exception:
            continue
    return parsed or dict(default)

__all__ = ["_flag", "_parse_budget_map", "_parse_csv_tuple", "_parse_float_map"]
