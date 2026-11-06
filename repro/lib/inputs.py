"""Fail before JIT/import-time model construction on incomplete research inputs."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from inventory import load_caps


def validate_inputs(data, caps):
    root = Path(data)
    paths = [root / "ds/dataset.npz", root / "ds/dataset_meta.csv", root / "calib/ag.csv", root / "calib/bd.csv", Path(caps)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required backtest inputs: " + ", ".join(missing))
    cap = load_caps(caps)
    meta = pd.read_csv(paths[1])
    required_meta = {"fut", "spot", "mult", "hs_s", "hs_f", "r2", "r5", "tick_s", "tick_f", "dl_exit", "dl_fin", "delisted", "grp"}
    if not required_meta.issubset(meta):
        raise ValueError("dataset metadata missing: " + ", ".join(sorted(required_meta-set(meta))))
    if meta.fut.duplicated().any(): raise ValueError("duplicate dataset symbols")
    if not (np.isfinite(meta.mult) & (meta.mult > 0)).all(): raise ValueError("invalid token multiplier")
    with np.load(paths[0], allow_pickle=False) as z:
        keys = {"sv", "sc", "sq", "fv", "fc", "fq", "ev", "r8", "r8s", "d02", "d1", "oi", "oih", "mk", "so", "sh", "sl", "fo", "fh", "fl"}
        if not keys.issubset(z.files): raise ValueError("dataset missing arrays: " + ", ".join(sorted(keys-set(z.files))))
        sc_all, fc_all = z["sc"], z["fc"]
        shape = sc_all.shape
        if len(shape) != 2 or shape[1] != len(meta): raise ValueError("dataset/metadata shape mismatch")
        if any(z[k].shape != shape for k in keys): raise ValueError("inconsistent dataset array shapes")
        ev, mk = z["ev"], z["mk"]
        if not np.isfinite(ev).all(): raise ValueError("nonfinite funding events")
        settled = ev != 0
        if "ivh" in z.files:
            ivh = z["ivh"]
            if ivh.shape != shape or not np.isfinite(ivh).all() or np.any((ivh < 0) | (ivh > 8) | (ivh != np.round(ivh))):
                raise ValueError("invalid explicit funding event intervals")
            if np.any((ev != 0) & (ivh <= 0)):
                raise ValueError("nonzero funding event has no explicit interval")
            settled = settled | (ivh > 0)
        eligible_cols = meta.fut.isin(cap.loc[cap.cap_qty_token > 0, "fut_symbol"]).to_numpy()
        if np.any(settled[:, eligible_cols] & (~np.isfinite(mk[:, eligible_cols]) | (mk[:, eligible_cols] <= 0))):
            raise ValueError("funding settlement marks missing for inventory symbols")
        # The engine quantity is raw tokens; futures prices must already be divided by mult.
        for j in range(len(meta)):
            sc, fc = sc_all[:, j], fc_all[:, j]
            ok = (sc > 0) & (fc > 0) & np.isfinite(sc) & np.isfinite(fc)
            if ok.any() and not 0.1 < np.median(fc[ok]/sc[ok]) < 10:
                raise ValueError("futures prices must be normalised to raw-token spot units")
    for path, required in [(paths[2], {"sym", "date", "y_kyle", "hs"}), (paths[3], {"sym", "date"})]:
        df = pd.read_csv(path)
        if not required.issubset(df): raise ValueError("invalid calibration columns: " + path.name)
    eligible = cap.loc[cap.cap_qty_token > 0, "fut_symbol"]
    if not eligible.isin(meta.fut).any(): raise ValueError("no positive inventory overlaps the dataset")
    hashes = {}
    for path in paths:
        with path.open("rb") as stream:
            hashes[str(path.relative_to(root)) if path.is_relative_to(root) else "private_caps"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(dataset_hours=shape[0], dataset_symbols=shape[1], inventory_symbols=len(cap),
                positive_inventory_symbols=len(eligible), covered_inventory_symbols=int(eligible.isin(meta.fut).sum()),
                uncovered_inventory_symbols=int((~eligible.isin(meta.fut)).sum()),
                quantity_unit="raw_token", inventory_history="static snapshot assumed available throughout; not historical proof",
                sha256=hashes)
