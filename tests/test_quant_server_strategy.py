import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from quant_server import StateStore, StrategyExecutor


class StrategyExecutorTest(unittest.TestCase):
    def test_cn_strategy_uses_cached_targets_when_data_layer_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp) / "state.json")
            executor = StrategyExecutor(store)

            result = asyncio.run(executor.run_strategy("CN"))

            self.assertEqual(result["status"], "success")
            json.dumps(result, allow_nan=False)
            self.assertEqual(store.state.strategies["CN"].status, "success")
            self.assertGreater(len(store.state.holdings["CN"]), 0)


if __name__ == "__main__":
    unittest.main()
