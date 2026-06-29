"""RQAlpha strategy that replays project-generated A-share target weights.

Generate weights first:

    python scripts/rqalpha_export_cn_targets.py --top-n 6 --rebalance 60

Then run this strategy with RQAlpha and set:

    RQALPHA_TARGET_WEIGHTS=data/rqalpha_bridge/cn_target_weights.csv

The strategy does not compute factors.  It only reads target weights and calls
``order_target_percent`` on rebalance dates, making RQAlpha an execution auditor.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from rqalpha.api import order_target_percent

TARGET_FILE = Path(os.environ.get(
    "RQALPHA_TARGET_WEIGHTS",
    "data/rqalpha_bridge/cn_target_weights.csv",
))


def init(context):
    df = pd.read_csv(TARGET_FILE)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    context.targets = {
        dt: g.set_index("order_book_id")["weight"].to_dict()
        for dt, g in df.groupby("date")
    }
    context.last_target = {}


def handle_bar(context, bar_dict):
    today = context.now.date()
    target = context.targets.get(today)
    if target is None:
        return

    # Sell names that disappeared from the target book.
    for order_book_id in list(context.last_target.keys()):
        if order_book_id not in target:
            order_target_percent(order_book_id, 0)

    # Then move current target names to requested weights.
    for order_book_id, weight in target.items():
        order_target_percent(order_book_id, weight)

    context.last_target = target
