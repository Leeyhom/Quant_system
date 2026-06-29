#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
飞书通知模块 - 量化策略运行后自动推送报告

配置方式：
  1. 在项目根目录新建 .env 文件，写入：
     FEISHU_WEBHOOK_URL = "<your-feishu-webhook-url>"
  2. 或者直接运行时设置环境变量

使用方式：
  python scripts/quant_engine.py --market CN --live --feishu
  python scripts/feishu_notify.py --market CN --test
"""
import os
import json
import requests
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """加载配置文件"""
    env_file = PROJECT_ROOT / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ[k.strip()] = v.strip()

    webhook = os.environ.get('FEISHU_WEBHOOK_URL', '')
    if not webhook:
        print("⚠️  未配置 FEISHU_WEBHOOK_URL，将跳过飞书通知")
        print("   请在 .env 文件中添加: FEISHU_WEBHOOK_URL = 你的webhook地址")
    return webhook


def send_portfolio_report(market: str, portfolio: dict, result: dict = None) -> bool:
    """推送持仓报告到飞书

    Args:
        market: CN/US/HK
        portfolio: 持仓数据（generate_live_portfolio返回）
        result: 回测结果（可选）
    """
    webhook = load_config()
    if not webhook:
        return False

    date = portfolio.get('date', datetime.now().strftime('%Y-%m-%d'))
    weights = portfolio.get('weights', {})
    factor_dirs = portfolio.get('factor_directions', {})
    factor_ics = portfolio.get('factor_ics', {})

    # 持仓表格
    holdings = []
    for i, (symbol, weight) in enumerate(sorted(weights.items(), key=lambda x: -x[1])):
        holdings.append(f"{i+1:2d}. {symbol:10s}  {weight:.1%}")

    # 因子方向表格
    factors = []
    for name, direction in factor_dirs.items():
        ic = factor_ics.get(name, 0)
        sign = "↗️" if direction == '+' else "↘️"
        factors.append(f"{name:15s} {sign} (IC={ic:+.4f})")

    title = f"📊 {market} 量化策略调仓报告 {date}"

    content = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": [
                        [{"tag": "text", "text": f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}],
                        [{"tag": "text", "text": f"📈 最新持仓 (TOP{len(holdings)} 等权):"}],
                        [{"tag": "text", "text": "\n".join(holdings[:10])}],
                    ]
                }
            }
        }
    }

    if len(holdings) > 10:
        content["content"]["post"]["zh_cn"]["content"].append([
            {"tag": "text", "text": f"  ... 还有 {len(holdings)-10} 只股票"}
        ])

    content["content"]["post"]["zh_cn"]["content"].extend([
        [{"tag": "text", "text": f"\n🧠 因子方向学习结果:"}],
        [{"tag": "text", "text": "\n".join(factors)}],
    ])

    # 回测表现（如果有）
    if result:
        s = result.get('strategy', {})
        sb = result.get('benchmark', {})
        content["content"]["post"]["zh_cn"]["content"].extend([
            [{"tag": "text", "text": f"\n📊 历史回测表现（Walk-Forward 无泄漏）:"}],
            [{"tag": "text", "text": (
                f"   策略夏普: {s.get('sharpe', 0):.2f}   |  累计收益: {s.get('total_return', 0):+.1%}\n"
                f"   基准夏普: {sb.get('sharpe', 0):.2f}   |  基准收益: {sb.get('total_return', 0):+.1%}\n"
                f"   超额夏普: {result.get('excess_sharpe', 0):+.2f}   |  最大回撤: {s.get('max_drawdown', 0):.1%}"
            )}],
        ])

    content["content"]["post"]["zh_cn"]["content"].append([
        {"tag": "text", "text": f"\n💡 提示: 以上仅为策略信号，不构成任何投资建议。"}
    ])

    try:
        resp = requests.post(webhook, json=content, timeout=10)
        resp.raise_for_status()
        print(f"✅ 飞书通知已发送")
        return True
    except Exception as e:
        print(f"❌ 飞书通知发送失败: {e}")
        return False


def send_alert(market: str, message: str, level: str = "info") -> bool:
    """发送告警信息

    level: info / warning / error
    """
    webhook = load_config()
    if not webhook:
        return False

    icons = {"info": "ℹ️", "warning": "⚠️", "error": "🚨"}
    icon = icons.get(level, "ℹ️")

    content = {
        "msg_type": "text",
        "content": {
            "text": f"{icon} [{market}] {message}"
        }
    }

    try:
        requests.post(webhook, json=content, timeout=10)
        return True
    except:
        return False


def send_daily_report(reports: dict) -> bool:
    """三市场收盘汇总日报

    reports: { 'CN': portfolio_dict, 'US': portfolio_dict, 'HK': portfolio_dict }
    """
    webhook = load_config()
    if not webhook:
        return False

    lines = [f"📅 {datetime.now().strftime('%Y-%m-%d')} 三市场量化策略日报\n"]

    for market, r in reports.items():
        market_icon = {"CN": "🇨🇳", "US": "🇺🇸", "HK": "🇭🇰"}.get(market, market)
        perf = r.get('performance', {})
        sharpe = perf.get('sharpe', 0)
        excess = perf.get('excess_sharpe', 0)
        status = "✅" if excess > 0 else "⏳"
        lines.append(f"{market_icon} {market}: 夏普{sharpe:.2f} 超额{excess:+.2f} {status}")

    lines.append(f"\n💡 完整持仓已保存到 results/ 目录")

    content = {
        "msg_type": "text",
        "content": {"text": "\n".join(lines)}
    }

    try:
        requests.post(webhook, json=content, timeout=10)
        return True
    except:
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', type=str, default='CN')
    parser.add_argument('--test', action='store_true', help='测试发送')
    args = parser.parse_args()

    if args.test:
        print("🧪 飞书通知测试...")
        # 伪造一份测试数据
        test_portfolio = {
            'date': '2025-12-31',
            'market': args.market,
            'weights': {f'STOCK{i:02d}': 0.05 for i in range(1, 21)},
            'factor_directions': {'factor1': '+', 'factor2': '-', 'factor3': '+'},
            'factor_ics': {'factor1': 0.03, 'factor2': -0.02, 'factor3': 0.01},
        }
        send_portfolio_report(args.market, test_portfolio)
