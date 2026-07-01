"""Run the RQAlpha replay strategy if rqalpha is installed.

This wrapper keeps the command reproducible, but it does not vendor or install
RQAlpha.  You still need an initialized RQAlpha data bundle.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_FILE = PROJECT_ROOT / "data" / "rqalpha_bridge" / "cn_target_weights.csv"
RESULT_FILE = PROJECT_ROOT / "data" / "rqalpha_bridge" / "rqalpha_result.pkl"
STRATEGY_FILE = PROJECT_ROOT / "scripts" / "rqalpha_cn_replay_strategy.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RQAlpha bridge replay")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--capital", type=float, default=60_000.0)
    p.add_argument("--bundle", default=None, help="optional RQAlpha bundle path")
    p.add_argument("--benchmark", default="000300.XSHG")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Missing target file: {TARGET_FILE}. Run rqalpha_export_cn_targets.py first.")
    if args.bundle is None:
        default_bundle = Path.home() / ".rqalpha" / "bundle"
        if not default_bundle.exists():
            raise SystemExit(
                "Missing RQAlpha data bundle: ~/.rqalpha/bundle. "
                "Run `rqalpha download-bundle` or pass `--bundle /path/to/bundle`."
            )

    cmd = [
        sys.executable, "-m", "rqalpha", "run",
        "-f", str(STRATEGY_FILE),
        "-s", args.start,
        "-e", args.end,
        "-bm", args.benchmark,
        "-o", str(RESULT_FILE),
        "--account", "stock", str(args.capital),
        "--stock-min-commission", "5",
        "--slippage", "0.0005",
        "--matching-type", "current_bar",
    ]
    if args.bundle:
        cmd.extend(["-d", args.bundle])

    env = os.environ.copy()
    env["RQALPHA_TARGET_WEIGHTS"] = str(TARGET_FILE)
    mpl_cache = PROJECT_ROOT / "data" / "matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    print(f"RQAlpha result: {RESULT_FILE}")


if __name__ == "__main__":
    main()
