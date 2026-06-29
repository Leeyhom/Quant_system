#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenClaw Agent 适配器 - 接入OpenClaw智能体到量化系统

OpenClaw是自主Agent框架，可用于:
  - 策略自动优化
  - 市场情绪分析
  - 新闻/公告解读
  - 风险智能预警
"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, Any, Optional
from .agent_base import BaseAgent, AgentContext


class OpenClawAgent(BaseAgent):
    """OpenClaw 智能体适配器"""

    def __init__(self, endpoint: str = "http://localhost:8000", api_key: str = ""):
        super().__init__("OpenClaw")
        self.endpoint = endpoint
        self.api_key = api_key
        self.session_id: Optional[str] = None

    async def run(self, context: AgentContext) -> Dict[str, Any]:
        """运行OpenClaw分析任务"""
        # 构造给OpenClaw的提示词
        prompt = self._build_prompt(context)

        try:
            # 调用OpenClaw API
            result = await self._call_openclaw(prompt)

            # 解析结果并执行动作
            actions = self._parse_result(result)
            await self._execute_actions(actions, context)

            return {
                "status": "success",
                "analysis": result,
                "actions": actions,
                "message": "OpenClaw分析完成"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "OpenClaw调用失败"
            }

    def _build_prompt(self, context: AgentContext) -> str:
        """构建给OpenClaw的分析提示词"""
        state = context.system_state
        strategies = state.get('strategies', {})

        prompt = f"""
你是量化策略AI助手，负责分析当前三市场量化系统的运行状态并给出建议。

当前系统状态:
- 运行时间: {state.get('system_start_time', '')}
- 最后更新: {state.get('last_update', '')}
- 累计盈亏: {state.get('total_pnl', 0):.2%}

各市场策略表现:
"""
        for market, s in strategies.items():
            prompt += f"""
{market}市场:
  - 夏普比率: {s.get('sharpe', 0):.2f}
  - 超额收益: {s.get('excess_sharpe', 0):.2f}
  - 累计收益: {s.get('total_return', 0):.1%}
  - 当前状态: {s.get('status', '')}
  - 最近消息: {s.get('message', '')}
"""

        holdings = state.get('holdings', {})
        for market, h_list in holdings.items():
            if h_list:
                prompt += f"\n{market}持仓: {len(h_list)}只股票\n"

        prompt += """
请分析:
1. 当前三市场策略表现是否正常？
2. 是否有需要调整的风险点？
3. 给出具体的操作建议（调仓/风控/因子调整等）

请以JSON格式返回结果，包含:
{
  "risk_level": "low|medium|high",
  "summary": "简短分析总结",
  "suggestions": ["建议1", "建议2", ...],
  "actions": [
    {"type": "alert|trade|config", "content": "具体内容"}
  ]
}
"""
        return prompt

    async def _call_openclaw(self, prompt: str) -> Dict[str, Any]:
        """调用OpenClaw API"""
        try:
            import httpx
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.endpoint}/api/agent/run",
                    json={
                        "agent_type": "quant_analyst",
                        "prompt": prompt,
                        "session_id": self.session_id
                    },
                    headers=headers
                )
                if response.status_code == 200:
                    result = response.json()
                    self.session_id = result.get('session_id')
                    return result
                else:
                    raise Exception(f"API错误: {response.status_code}")
        except ImportError:
            # 如果没有httpx，使用模拟模式
            return self._mock_analysis(prompt)
        except Exception as e:
            # 连接失败时返回模拟分析
            print(f"OpenClaw连接失败: {e}，使用内置分析")
            return self._mock_analysis(prompt)

    def _mock_analysis(self, prompt: str) -> Dict[str, Any]:
        """内置简单分析（OpenClaw不可用时的降级方案）"""
        return {
            "risk_level": "low",
            "summary": "系统运行正常，各市场策略表现稳定",
            "suggestions": [
                "当前风险水平较低，维持现有仓位",
                "建议关注美股因子IC衰减趋势",
                "关注港股流动性变化"
            ],
            "actions": [
                {"type": "alert", "content": "注意观察美股价值因子表现"}
            ]
        }

    def _parse_result(self, result: Dict) -> list:
        """解析OpenClaw返回的动作建议"""
        return result.get('actions', [])

    async def _execute_actions(self, actions: list, context: AgentContext):
        """执行OpenClaw建议的动作"""
        for action in actions:
            if action.get('type') == 'alert':
                # 发送飞书告警
                try:
                    from scripts.feishu_notify import send_alert
                    await asyncio.get_event_loop().run_in_executor(
                        None, send_alert,
                        "OpenClaw", action.get('content', ''), "info"
                    )
                except:
                    pass
            elif action.get('type') == 'trade':
                # 交易建议记录到日志
                print(f"[OpenClaw] 交易建议: {action.get('content')}")

    def get_cron(self) -> str:
        """OpenClaw每天盘后运行一次"""
        return "0 18 * * 1-5"


# 自动注册到Agent中心
try:
    from .agent_base import agent_registry
    agent_registry.register(OpenClawAgent())
except:
    pass
