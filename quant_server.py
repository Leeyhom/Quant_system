#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化策略实时服务引擎

功能架构：
  - HTTP/WebSocket 服务：实时看板UI
  - 调度器：定时运行策略/因子更新
  - 状态持久化：持仓/PnL/因子状态
  - 飞书Bot：告警/日报/调仓通知
  - Agent插件接口：支持OpenClaw等智能体接入

运行方式：
  python quant_server.py --port 8080
  访问 http://localhost:8080 查看实时看板
"""
from __future__ import annotations

import os
import sys
import json
import asyncio
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field

# 第三方库
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("⚠️  请安装 FastAPI: pip install fastapi uvicorn")
    sys.exit(1)

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:
    print("⚠️  请安装 APScheduler: pip install apscheduler")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# =====================================================================
# 数据模型
# =====================================================================
@dataclass
class StrategyState:
    """策略运行状态"""
    market: str
    last_run: str = ""
    next_run: str = ""
    status: str = "idle"  # idle / running / error
    sharpe: float = 0.0
    excess_sharpe: float = 0.0
    total_return: float = 0.0
    current_pnl: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    message: str = ""


@dataclass
class Holding:
    """持仓记录"""
    symbol: str
    weight: float
    entry_date: str
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    pnl: float = 0.0


@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    symbol: str
    action: str  # BUY / SELL / ADJUST
    quantity: float
    price: float
    reason: str = ""


@dataclass
class SystemState:
    """系统全局状态"""
    strategies: Dict[str, StrategyState] = field(default_factory=dict)
    holdings: Dict[str, List[Holding]] = field(default_factory=dict)
    trades: List[TradeRecord] = field(default_factory=list)
    factor_states: Dict[str, Dict] = field(default_factory=dict)
    system_start_time: str = ""
    last_update: str = ""
    total_pnl: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "strategies": {k: asdict(v) for k, v in self.strategies.items()},
            "holdings": {k: [asdict(h) for h in v] for k, v in self.holdings.items()},
            "trades": [asdict(t) for t in self.trades],
            "factor_states": self.factor_states,
            "system_start_time": self.system_start_time,
            "last_update": self.last_update,
            "total_pnl": self.total_pnl,
        }


# =====================================================================
# 状态持久化
# =====================================================================
class StateStore:
    """状态存储器（SQLite简单实现，后续可扩展Redis）"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = SystemState()
        self.load()

    def load(self):
        """从磁盘加载状态"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.state.system_start_time = data.get('system_start_time', '')
                    self.state.last_update = data.get('last_update', '')
                    self.state.total_pnl = data.get('total_pnl', 0.0)
                    for m, s in data.get('strategies', {}).items():
                        self.state.strategies[m] = StrategyState(**s)
                    for m, h in data.get('holdings', {}).items():
                        self.state.holdings[m] = [Holding(**x) for x in h]
                    for t in data.get('trades', []):
                        self.state.trades.append(TradeRecord(**t))
                    self.state.factor_states = data.get('factor_states', {})
            except Exception as e:
                print(f"状态加载失败: {e}，使用全新状态")
        if not self.state.system_start_time:
            self.state.system_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def save(self):
        """保存状态到磁盘"""
        self.state.last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)

    def update_strategy(self, market: str, **kwargs):
        """更新策略状态"""
        if market not in self.state.strategies:
            self.state.strategies[market] = StrategyState(market=market)
        for k, v in kwargs.items():
            if hasattr(self.state.strategies[market], k):
                setattr(self.state.strategies[market], k, v)
        self.save()

    def add_trade(self, trade: TradeRecord):
        """记录交易"""
        self.state.trades.insert(0, trade)
        if len(self.state.trades) > 1000:  # 只保留最近1000条
            self.state.trades = self.state.trades[:1000]
        self.save()

    def update_holdings(self, market: str, holdings: List[Holding]):
        """更新持仓"""
        self.state.holdings[market] = holdings
        self.save()


# =====================================================================
# 策略执行引擎
# =====================================================================
class StrategyExecutor:
    """策略执行器"""

    def __init__(self, store: StateStore):
        self.store = store
        self.running = False

    async def run_strategy(self, market: str, is_live: bool = True) -> Dict:
        """运行单个市场策略"""
        print(f"[{datetime.now()}] 运行 {market} 策略...")
        self.store.update_strategy(market, status="running", message="正在计算因子...")

        try:
            from scripts.quant_engine import generate_live_portfolio, walk_forward_backtest, load_factors
            from quant.backtest.metrics import summary

            # 生成最新持仓
            portfolio = generate_live_portfolio(market)

            # 解析持仓
            holdings = [
                Holding(symbol=sym, weight=w, entry_date=portfolio['date'])
                for sym, w in portfolio['weights'].items()
            ]
            self.store.update_holdings(market, holdings)
            self.store.state.factor_states[market] = portfolio.get('factor_ics', {})

            # 计算策略表现（全样本回测）
            self.store.update_strategy(
                market,
                last_run=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                status="success",
                message="策略运行完成",
                sharpe=portfolio.get('sharpe', 0),
                excess_sharpe=portfolio.get('excess_sharpe', 0),
            )

            # 飞书通知（如果开启）
            if os.environ.get('FEISHU_NOTIFY', '0') == '1':
                from scripts.feishu_notify import send_portfolio_report
                await asyncio.get_event_loop().run_in_executor(
                    None, send_portfolio_report, market, portfolio
                )

            print(f"[{datetime.now()}] {market} 策略运行完成")
            return {"status": "success", "portfolio": portfolio}

        except Exception as e:
            import traceback
            error_msg = f"策略运行失败: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            self.store.update_strategy(market, status="error", message=error_msg)
            return {"status": "error", "error": error_msg}

    async def run_all_strategies(self):
        """运行所有市场策略"""
        results = {}
        for market in ['CN', 'US', 'HK']:
            results[market] = await self.run_strategy(market)
        return results

    async def send_daily_report(self):
        """发送日报"""
        from scripts.feishu_notify import send_daily_report
        reports = {}
        for m in ['CN', 'US', 'HK']:
            if m in self.store.state.strategies:
                s = self.store.state.strategies[m]
                reports[m] = {
                    'performance': {
                        'sharpe': s.sharpe,
                        'excess_sharpe': s.excess_sharpe,
                        'total_return': s.total_return,
                    }
                }
        await asyncio.get_event_loop().run_in_executor(None, send_daily_report, reports)


# =====================================================================
# WebSocket连接管理器
# =====================================================================
class ConnectionManager:
    """WebSocket连接管理器 - 实时推送更新到前端"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        """广播消息给所有连接的客户端"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


# =====================================================================
# Agent插件接口
# =====================================================================
class AgentPlugin:
    """Agent插件基类，用于接入OpenClaw等智能体"""

    def __init__(self, name: str):
        self.name = name

    async def run(self, context: Dict) -> Dict:
        """运行Agent"""
        raise NotImplementedError

    def get_schedule(self) -> str:
        """返回cron表达式"""
        return "0 0 * * *"  # 默认每天0点运行


class SimpleMonitorAgent(AgentPlugin):
    """简单监控Agent示例"""

    def __init__(self):
        super().__init__("MonitorAgent")

    async def run(self, context: Dict) -> Dict:
        """监控告警：回撤过大/单票波动异常"""
        state = context.get('state', {})
        alerts = []

        for m, s in state.get('strategies', {}).items():
            if s.get('status') == 'error':
                alerts.append(f"[{m}] 策略运行异常: {s.get('message', '')}")

        if alerts:
            from scripts.feishu_notify import send_alert
            for a in alerts:
                send_alert("ALL", a, level="warning")

        return {"alerts": alerts}


# =====================================================================
# FastAPI 应用
# =====================================================================
def create_app(store: StateStore, executor: StrategyExecutor) -> FastAPI:
    """创建Web应用"""
    app = FastAPI(title="量化策略实时服务", version="2.0")
    manager = ConnectionManager()
    scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

    # 静态资源目录
    static_dir = PROJECT_ROOT / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # =================================================================
    # HTTP 接口
    # =================================================================
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """实时看板主页"""
        html_path = PROJECT_ROOT / "templates" / "realtime_dashboard.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding='utf-8'))
        return HTMLResponse("<h1>量化服务运行中...请访问 /docs 查看API</h1>")

    @app.get("/api/state")
    async def get_state():
        """获取系统全局状态"""
        return {"code": 0, "data": store.state.to_dict()}

    @app.get("/api/strategy/{market}")
    async def get_strategy(market: str):
        """获取单个策略状态"""
        if market in store.state.strategies:
            return {"code": 0, "data": asdict(store.state.strategies[market])}
        return {"code": 404, "error": "策略不存在"}

    @app.get("/api/holdings/{market}")
    async def get_holdings(market: str):
        """获取当前持仓"""
        return {"code": 0, "data": [asdict(h) for h in store.state.holdings.get(market, [])]}

    @app.get("/api/trades")
    async def get_trades(limit: int = 50):
        """获取交易记录"""
        return {"code": 0, "data": [asdict(t) for t in store.state.trades[:limit]]}

    @app.post("/api/run/{market}")
    async def run_strategy_now(market: str):
        """立即运行策略"""
        result = await executor.run_strategy(market)
        await manager.broadcast({"type": "state_update", "data": store.state.to_dict()})
        return result

    @app.post("/api/run-all")
    async def run_all_now():
        """立即运行所有策略"""
        results = await executor.run_all_strategies()
        await manager.broadcast({"type": "state_update", "data": store.state.to_dict()})
        return {"code": 0, "results": results}

    # =================================================================
    # WebSocket 实时推送
    # =================================================================
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            # 首次连接发送全量状态
            await websocket.send_json({"type": "init", "data": store.state.to_dict()})
            while True:
                # 保持连接，等待心跳
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    # =================================================================
    # 定时任务
    # =================================================================
    @app.on_event("startup")
    async def startup():
        """服务启动"""
        print("=" * 60)
        print("🚀 量化策略实时服务启动中...")
        print("=" * 60)

        # 设置定时任务（A股/港股收盘后运行）
        scheduler.add_job(
            executor.run_strategy,
            trigger=CronTrigger(hour=15, minute=30, day_of_week='mon-fri', timezone='Asia/Shanghai'),
            args=['CN']
        )
        scheduler.add_job(
            executor.run_strategy,
            trigger=CronTrigger(hour=16, minute=30, day_of_week='mon-fri', timezone='Asia/Shanghai'),
            args=['HK']
        )
        scheduler.add_job(
            executor.run_strategy,
            trigger=CronTrigger(hour=5, minute=30, day_of_week='mon-fri', timezone='Asia/Shanghai'),
            args=['US']
        )
        # 每天收盘后发送日报
        scheduler.add_job(
            executor.send_daily_report,
            trigger=CronTrigger(hour=17, minute=0, day_of_week='mon-fri', timezone='Asia/Shanghai')
        )

        scheduler.start()
        print(f"📅 定时任务已启动: A股15:30 / 港股16:30 / 美股05:30")
        print(f"🌐 访问地址: http://localhost:{os.environ.get('PORT', 8080)}")
        print(f"📊 API文档: http://localhost:{os.environ.get('PORT', 8080)}/docs")

        # 启动时先运行一次所有策略
        asyncio.create_task(executor.run_all_strategies())

    @app.on_event("shutdown")
    async def shutdown():
        """服务关闭"""
        scheduler.shutdown()
        print("👋 服务已停止")

    return app


# =====================================================================
# 主入口
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="量化策略实时服务")
    parser.add_argument('--port', type=int, default=8080, help='服务端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--env', type=str, default='', help='环境配置文件路径')
    parser.add_argument('--feishu', action='store_true', help='开启飞书通知')
    args = parser.parse_args()

    # 加载环境变量
    if args.env and Path(args.env).exists():
        with open(args.env, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ[k.strip()] = v.strip()

    if args.feishu:
        os.environ['FEISHU_NOTIFY'] = '1'

    os.environ['PORT'] = str(args.port)

    # 初始化组件
    store = StateStore(PROJECT_ROOT / "data" / "system_state.json")
    executor = StrategyExecutor(store)

    # 创建应用
    app = create_app(store, executor)

    # 启动服务
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == '__main__':
    main()
