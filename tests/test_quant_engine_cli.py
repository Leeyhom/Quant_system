import subprocess
import sys
import unittest


class QuantEngineCliTest(unittest.TestCase):
    def test_cn_live_cli_handles_cached_targets_without_prices(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/quant_engine.py",
                "--market",
                "CN",
                "--live",
            ],
            text=True,
            capture_output=True,
            encoding="utf-8",
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("实盘持仓生成", result.stdout)


if __name__ == "__main__":
    unittest.main()
