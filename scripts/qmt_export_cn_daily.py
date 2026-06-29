"""Export A-share daily bars from QMT/xtquant into this project's cache layout.

Run this on the machine where QMT/xtquant is installed and logged in:

    python scripts/qmt_export_cn_daily.py --pool default --start 20180101 --end 20251231

Output:

    data/qmt/daily/600519.parquet   # preferred
    data/qmt/daily/000001.parquet   # falls back to .csv if parquet is unavailable
    ...

Each parquet uses the same columns as ``quant.data.akshare_loader``:

    date, open, high, low, close, volume, amount

The first A/B pass should keep the same DEFAULT_POOL and same strategy logic;
only switch the loader from akshare to QMT.  Trade-status/limit-up/limit-down
constraints can be layered in the next pass once daily bars match cleanly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant.config import DATA_DIR, HISTORY_END, HISTORY_START
from quant.data.universe import DEFAULT_POOL
from quant.data.qmt_loader import save_parquet

QMT_DAILY_DIR = DATA_DIR / "qmt" / "daily"


def to_qmt_symbol(symbol: str) -> str:
    """Convert project 6-digit A-share code to QMT's exchange suffix format."""
    s = str(symbol).strip().upper()
    if "." in s:
        return s
    if s.startswith(("6", "9")):
        return f"{s}.SH"
    return f"{s}.SZ"


def to_project_symbol(symbol: str) -> str:
    """Convert ``600519.SH`` / ``000001.SZ`` back to the 6-digit project code."""
    return str(symbol).strip().upper().split(".")[0]


def _extract_field_frame(raw, field: str) -> pd.DataFrame:
    """Handle common xtdata return shapes for one field.

    xtquant versions differ a little: some return ``dict[field]`` DataFrames,
    some wrap stock/date orientation differently.  This helper normalizes enough
    to keep the export script resilient.
    """
    if isinstance(raw, dict):
        if field not in raw:
            raise KeyError(f"QMT result missing field: {field}")
        obj = raw[field]
    else:
        obj = raw

    if not isinstance(obj, pd.DataFrame):
        obj = pd.DataFrame(obj)
    return obj.copy()


def _series_from_field(field_df: pd.DataFrame, qmt_symbol: str) -> pd.Series:
    """Pick one stock's time series from a field DataFrame."""
    if qmt_symbol in field_df.columns:
        s = field_df[qmt_symbol]
    elif qmt_symbol in field_df.index:
        s = field_df.loc[qmt_symbol]
    else:
        plain = to_project_symbol(qmt_symbol)
        if plain in field_df.columns:
            s = field_df[plain]
        elif plain in field_df.index:
            s = field_df.loc[plain]
        elif field_df.shape[1] == 1:
            s = field_df.iloc[:, 0]
        elif field_df.shape[0] == 1:
            s = field_df.iloc[0]
        else:
            raise KeyError(f"Cannot locate {qmt_symbol} in QMT field frame")

    s = pd.Series(s).copy()
    s.index = pd.to_datetime(s.index.astype(str), errors="coerce")
    s = s[~s.index.isna()]
    s = pd.to_numeric(s, errors="coerce")
    return s.sort_index()


def _get_market_data_one(xtdata, qmt_symbol: str, start: str, end: str, dividend_type: str) -> pd.DataFrame:
    """Download/read one stock's daily OHLCV from xtdata."""
    # Download first so get_market_data can read from the local QMT cache.  Some
    # xtquant builds require start/end as YYYYMMDD strings.
    try:
        xtdata.download_history_data(
            qmt_symbol,
            period="1d",
            start_time=start,
            end_time=end,
            incrementally=True,
        )
    except TypeError:
        # Older builds may not support ``incrementally``.
        xtdata.download_history_data(qmt_symbol, period="1d", start_time=start, end_time=end)

    fields = ["open", "high", "low", "close", "volume", "amount"]
    raw = xtdata.get_market_data(
        field_list=fields,
        stock_list=[qmt_symbol],
        period="1d",
        start_time=start,
        end_time=end,
        count=-1,
        dividend_type=dividend_type,
        fill_data=False,
    )

    series = {}
    for field in fields:
        frame = _extract_field_frame(raw, field)
        series[field] = _series_from_field(frame, qmt_symbol)

    df = pd.DataFrame(series).dropna(subset=["close"], how="all")
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", *fields]].sort_values("date").reset_index(drop=True)


def _load_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [to_project_symbol(s) for s in args.symbols]
    if args.symbol_file:
        p = Path(args.symbol_file)
        return [to_project_symbol(x.strip()) for x in p.read_text().splitlines() if x.strip()]
    if args.pool == "default":
        return list(DEFAULT_POOL)
    raise ValueError("No symbols selected")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export QMT A-share daily bars to data/qmt/daily")
    p.add_argument("--pool", choices=["default"], default="default", help="stock universe to export")
    p.add_argument("--symbols", nargs="*", help="explicit symbols, e.g. 600519 000001 or 600519.SH")
    p.add_argument("--symbol-file", help="one symbol per line")
    p.add_argument("--start", default=HISTORY_START)
    p.add_argument("--end", default=HISTORY_END)
    p.add_argument(
        "--dividend-type",
        default="front",
        help="QMT dividend_type passed to xtdata.get_market_data; use your QMT-supported qfq/front value",
    )
    p.add_argument("--limit", type=int, default=None, help="export first N symbols only, useful for smoke tests")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from xtquant import xtdata
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "xtquant is not available. Run this script inside the QMT Python environment "
            "on the machine where QMT is installed and logged in."
        ) from exc

    symbols = _load_symbols(args)
    if args.limit:
        symbols = symbols[: args.limit]
    QMT_DAILY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Export QMT daily bars: {len(symbols)} symbols | {args.start}~{args.end}")
    print(f"Output: {QMT_DAILY_DIR}")

    ok, failed = 0, []
    for i, sym in enumerate(symbols, 1):
        qmt_sym = to_qmt_symbol(sym)
        try:
            df = _get_market_data_one(xtdata, qmt_sym, args.start, args.end, args.dividend_type)
            if df.empty:
                raise ValueError("empty daily data")
            save_parquet(df, sym)
            ok += 1
            print(f"  [{i:3d}/{len(symbols)}] {sym} <- {qmt_sym}  OK  {len(df)} rows")
        except Exception as exc:  # noqa: BLE001
            failed.append((sym, type(exc).__name__, str(exc)))
            print(f"  [{i:3d}/{len(symbols)}] {sym} <- {qmt_sym}  FAIL  {type(exc).__name__}: {exc}")

    print(f"\nDone. success={ok}, failed={len(failed)}")
    if failed:
        fail_path = DATA_DIR / "qmt" / "export_failures.csv"
        pd.DataFrame(failed, columns=["symbol", "error_type", "error"]).to_csv(fail_path, index=False)
        print(f"Failure list: {fail_path}")


if __name__ == "__main__":
    main()
