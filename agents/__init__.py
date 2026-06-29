# Agent插件系统
from .agent_base import (
    BaseAgent,
    AgentContext,
    AgentRegistry,
    agent_registry,
    RiskMonitorAgent,
    PnLReportAgent,
    FactorRotationAgent,
)

try:
    from .openclaw_agent import OpenClawAgent
except ImportError:
    pass

__all__ = [
    'BaseAgent',
    'AgentContext',
    'AgentRegistry',
    'agent_registry',
    'RiskMonitorAgent',
    'PnLReportAgent',
    'FactorRotationAgent',
    'OpenClawAgent',
]
