"""Private caps are raw-token quantities, matching spot-equivalent engine prices."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_caps(path):
    frame = pd.read_csv(path, dtype={"fut_symbol": str})
    required = {"fut_symbol", "cap_usdt", "cap_qty_token"}
    if not required.issubset(frame):
        raise ValueError("cap CSV requires fut_symbol,cap_usdt,cap_qty_token")
    if frame.fut_symbol.isna().any() or frame.fut_symbol.str.strip().eq("").any():
        raise ValueError("empty inventory symbol")
    frame["fut_symbol"] = frame.fut_symbol.str.strip()
    if frame.fut_symbol.duplicated().any():
        raise ValueError("duplicate inventory symbols; reconcile borrowing sources first")
    for name in ("cap_usdt", "cap_qty_token"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if (~np.isfinite(frame[name]) | (frame[name] < 0)).any():
            raise ValueError(f"{name} must contain finite nonnegative numbers")
    return frame


def convert_inventory(source, destination, capital=10_000_000):
    """Read user's already-discounted 30% TSV. Never discount again or mark at future prices.

    cap_usdt is a configurable ceiling, NOT an inventory valuation or promised liquidity.
    cap_qty_token remains raw tokens; base quantity = tokens / multiplier is audit only.
    """
    from decimal import Decimal, InvalidOperation
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be positive")
    frame = pd.read_csv(source, sep="\t", dtype=str).fillna("")
    cols = ["Token", "Borrowable Qty @ 30% (Token)", "Binance Has Contract", "Binance Symbols",
            "BN Multiplier (Token/Base)", "BN Borrowable Qty (Contract Base)"]
    if not set(cols).issubset(frame):
        raise ValueError("unrecognised inventory TSV headers")
    def number(s):
        try:
            x = Decimal(s.strip().replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError("invalid inventory number") from exc
        if not x.is_finite() or x < 0:
            raise ValueError("inventory numbers must be finite and nonnegative")
        return x
    records = []
    for row in frame.to_dict("records"):
        qty = number(row[cols[1]])
        if row[cols[2]].strip().upper() != "Y":
            continue
        symbol = row[cols[3]].strip()
        mult = number(row[cols[4]])
        if not symbol.endswith("USDT") or mult <= 0:
            raise ValueError("ambiguous Binance contract mapping")
        # Both displayed columns have variable decimal precision; allow their rounding.
        supplied = number(row[cols[5]])
        tolerance = Decimal("0.51") * (Decimal(10) ** supplied.as_tuple().exponent + Decimal(10) ** qty.as_tuple().exponent / mult)
        if abs(supplied - qty / mult) > tolerance:
            raise ValueError("token/base quantity disagrees with multiplier")
        records.append(dict(fut_symbol=symbol, cap_usdt=capital, cap_qty_token=str(qty)))
    out = pd.DataFrame(records, columns=["fut_symbol", "cap_usdt", "cap_qty_token"])
    if out.fut_symbol.duplicated().any():
        raise ValueError("duplicate Binance mapping; manual reconciliation required")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)
    destination.chmod(0o600)
    load_caps(destination)
    return dict(source_rows=len(frame), mapped_symbols=len(out), positive_symbols=int((out.cap_qty_token.astype(float) > 0).sum()),
                quantity_unit="raw_token", inventory_discount_already_applied=True, borrow_interest=0,
                capital_usdt=capital, annualization_denominator_usdt=capital)
