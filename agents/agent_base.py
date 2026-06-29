#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent插件系统 - 支持OpenClaw等智能体接入

每个Agent实现run()方法，返回执行结果。
调度器负责按cron时间运行各Agent。
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class AgentContext:
    """Agent运行上下文 - 提供系统状态给智能体"""
    system_state: Dict[str, Any]
    strategy_results: Dict[str, Any]
    market_data: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Agent基类 - 所有自定义Agent必须继承此类"""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    @abstractmethod
    async def run(self, context: AgentContext) -> Dict[str, Any]:
        """
        运行Agent核心逻辑

        Args:
            context: 系统运行上下文，包含策略状态/市场数据/配置等

        Returns:
            执行结果字典，包含status/message/actions等字段
        """
        raise NotImplementedError

    def get_cron(self) -> str:
        """
        返回Agent运行的cron表达式
        例如: "0 15 * * 1-5" = 每个工作日15点运行
        """
        return "0 15 * * 1-5"  # 默认A股收盘后运行

    def get_name(self) -> str:
        return self.name


# =====================================================================
# 示例Agent实现
# =====================================================================
class RiskMonitorAgent(BaseAgent):
    """风险监控Agent - 监控回撤/波动/单票风险"""

    def __init__(self):
        super().__init__("RiskMonitor")

    async def run(self, context: AgentContext) -> Dict[str, Any]:
        alerts = []
        state = context.system_state

        # 检查各市场最大回撤
        for market, strategy in state.get('strategies', {}).items():
            max_dd = strategy.get('max_drawdown', 0)
            if abs(max_dd) > 0.15:  # 回撤>15%告警
                alerts.append(f"[{market}] 最大回撤预警: {max_dd:.1%}")

            if strategy.get('status') == 'error':
                alerts.append(f"[{market}] 策略运行异常: {strategy.get('message', '')}")

        # 检查单票集中度
        for market, holdings in state.get('holdings', {}).items():
            weights = [h.get('weight', 0) for h in holdings]
            if weights and max(weights) > 0.15:  # 单票权重>15%
                alerts.append(f"[{market}] 单票权重集中度预警")

        result = {
            "status": "success" if len(alerts) == 0 else "warning",
            "alerts": alerts,
            "message": f"风险检查完成，发现{len(alerts)}项告警"
        }

        # 发送告警
        if alerts:
            try:
                from scripts.feishu_notify import send_alert
                for alert in alerts:
                    await asyncio.get_event_loop().run_in_executor(
                        None, send_alert, "ALL", alert, "warning"
                    )
            except:
                pass

        return result

    def get_cron(self) -> str:
        return "*/30 * * * 1-5"  # 每30分钟运行一次


class PnLReportAgent(BaseAgent):
    """日报Agent - 统计每日盈亏并推送报告"""

    def __init__(self):
        super().__init__("PnLReporter")

    async def run(self, context: AgentContext) -> Dict[str, Any]:
        # 计算当日/累计收益
        state = context.system_state
        daily_pnl = state.get('daily_pnl', 0)
        total_pnl = state.get('total_pnl', 0)

        report = {
            "date": context.config.get('current_date'),
            "daily_pnl": daily_pnl,
            "total_pnl": total_pnl,
            "strategy_summary": {
                m: {
                    "sharpe": s.get('sharpe', 0),
                    "return": s.get('total_return', 0)
                }
                for m, s in state.get('strategies', {}).items()
            }
        }

        return {
            "status": "success",
            "report": report,
            "message": "日报生成完成"
        }

    def get_cron(self) -> str:
        return "0 17 * * 1-5"  # 每天17点发送日报


class FactorRotationAgent(BaseAgent):
    """因子轮动Agent - 基于因子表现动态调整权重"""

    def __init__(self):
        super().__init__("FactorRotation")

    async def run(self, context: AgentContext) -> Dict[str, Any]:
        """根据最近IC表现调整因子权重"""
        factor_ics = context.system_state.get('factor_states', {})
        actions = []

        # 简单的因子轮动逻辑: 只保留IC>0.02的强因子
        for market, factors in factor_ics.items():
            strong_factors = [name for name, ic in factors.items() if ic > 0.02]
            weak_factors = [name for name, ic in factors.items() if ic < -0.01]

            if strong_factors:
                actions.append(f"[{market}] 建议加仓因子: {', '.join(strong_factors)}")
            if weak_factors:
                actions.append(f"[{market}] 建议减仓因子: {', '.join(weak_factors)}")

        return {
            "status": "success",
            "actions": actions,
            "message": f"因子轮动分析完成，建议{len(actions)}项调整"
        }

    def get_cron(self) -> str:
        return "0 0 * * 0"  # 每周日运行


# =====================================================================
# Agent注册中心
# =====================================================================
class AgentRegistry:
    """Agent注册中心 - 管理所有可用Agent"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册内置Agent"""
        self.register(RiskMonitorAgent())
        self.register(PnLReportAgent())
        self.register(FactorRotationAgent())

    def register(self, agent: BaseAgent):
        """注册新Agent"""
        self.agents[agent.get_name()] = agent
        print(f"✅ Agent已注册: {agent.get_name()}")

    def unregister(self, name: str):
        """注销Agent"""
        if name in self.agents:
            del self.agents[name]
            print(f"♻️  Agent已注销: {name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """获取Agent实例"""
        return self.agents.get(name)

    def list_agents(self) -> Dict[str, str]:
        """列出所有Agent及其cron"""
        return {name: agent.get_cron() for name, agent in self.agents.items()}


# 全局Agent注册中心
agent_registry = AgentRegistry()
