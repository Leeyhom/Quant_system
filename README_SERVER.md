# 量化策略实时服务 v2.0

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    quant_server.py                              │
├──────────┬───────────┬───────────────────────────────┤
│  Web服务  │ 调度器   │   Agent插件系统              │
│  +UI    │  Cron    │   ┌─────────────────────┐    │
│  +API    │  定时运行│   │ OpenClaw 智能体  │    │
│  +WS     │           │   └─────────────────────┘    │
└──────────┴───────────┴───────────────────────────────┘
          │                        │
          ▼                        ▼
    三市场策略引擎          飞书Bot通知
          │
    ┌────────┴────────┐
    │ 状态持久化       │
    │ StateStore       │
    │ (JSON/SQLite)   │
    └──────────────────┘
```

---

## 快速开始

### 1. 安装依赖

```bash
# 基础量化依赖（已有）
pip install -r requirements.txt

# 实时服务依赖
pip install -r requirements-server.txt
```

### 2. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入飞书机器人地址等
vim .env
```

### 3. 启动服务

```bash
chmod +x start_server.sh

# 方式1: 使用启动脚本
./start_server.sh

# 方式2: 直接运行
python quant_server.py --port 8080 --feishu
```

---

## 功能特性

### 🌐 Web实时看板

访问: http://localhost:8080

功能:
- 📊 三市场策略实时状态
- 📈 净值/回撤/夏普等指标
- 💹 当前持仓与交易记录
- 🧠 因子IC状态监控
- 📋 运行日志实时流

### 🤖 Agent插件系统

内置Agent:
- **RiskMonitorAgent**: 风险监控（每30分钟检查一次，回撤超限告警）
- **PnLReportAgent**: 日报推送（每日17点）
- **FactorRotationAgent**: 因子轮动分析（每周日）
- **OpenClawAgent**: OpenClaw智能体接入（每日18点）

### 📅 自动交易调度

| 市场 | 运行时间 | 说明 |
|------|---------|------|
| A股 | 15:30 (周一至周五) | 收盘后计算最新持仓 |
| 港股 | 16:30 (周一至周五) | 收盘后计算最新持仓 |
| 美股 | 05:30 (周一至周五) | 前一日收盘后计算 |
| 日报 | 17:00 (周一至周五) | 三市场汇总报告 |

### 🔔 飞书通知

开启方式:
```bash
# 启动时开启
python quant_server.py --feishu

# 或在 .env 中设置
FEISHU_NOTIFY=1
```

通知内容:
- ✅ 持仓调整通知
- 📊 策略运行报告
- ⚠️ 风险监控告警
- 📈 每日盈亏日报

---

## API接口文档

访问 `http://localhost:8080/docs` 查看完整API文档。

### 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/state` | 获取全局状态 |
| GET | `/api/strategy/{market}` | 获取策略状态 |
| GET | `/api/holdings/{market}` | 获取当前持仓 |
| GET | `/api/trades` | 获取交易记录 |
| POST | `/api/run/{market}` | 立即运行策略 |
| POST | `/api/run-all` | 运行全部策略 |
| WS | `/ws` | WebSocket实时推送 |

---

## Agent开发

### 编写自定义Agent

```python
from agents import BaseAgent, AgentContext

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__("MyAgent")

    async def run(self, context: AgentContext) -> dict:
        # 你的自定义逻辑
        return {
            "status": "success",
            "message": "执行完成"
        }

    def get_cron(self) -> str:
        return "0 * * * *"  # 每小时运行一次
```

### 注册Agent:

```python
from agents import agent_registry
agent_registry.register(MyAgent())
```

---

## 项目结构

```
.
├── quant_server.py          # 服务主入口
├── quant_engine.py          # 策略引擎v2（嵌入到quant_server.py）
├── start_server.sh          # 启动脚本
├── .env.example             # 配置模板
├── requirements-server.txt  # 服务依赖
├── templates/               # 前端模板
│   └── realtime_dashboard.html
├── agents/                  # Agent插件
│   ├── __init__.py
│   ├── agent_base.py       # Agent基类
│   └── openclaw_agent.py # OpenClaw适配器
└── data/
│   └── system_state.json   # 状态持久化文件
```

---

## 常见问题

### Q: 如何接入实盘交易？

当前版本为策略计算与通知功能，实盘交易需要对接券商API：

1. 选择券商（盈透/老虎/富途等）
2. 继承 `BaseBroker` 类实现下单接口
3. 配置 `.env` 中设置 `TRADING_MODE=LIVE`
4. 完成模拟盘运行至少1个月验证稳定性

### Q: 如何自定义策略参数？

策略参数在 `scripts/quant_engine.py` 中通过命令行参数传入，或直接在 `.env` 配置。

### Q: 如何接入OpenClaw？

1. 启动OpenClaw服务
2. 在 `.env` 中配置:
```
OPENCLAW_ENDPOINT=http://openclaw-address:8000
OPENCLAW_API_KEY=
```

### Q: 数据持久化存储？

当前使用JSON文件简单持久化，生产环境建议迁移到：
- SQLite（推荐，零依赖）
- Redis（高性能读写）
- PostgreSQL（完整数据归档）

---

## 版本历史

### v2.0（当前）
- ✅ 实时WebSocket推送
- ✅ 三市场统一调度
- ✅ Agent插件系统
- ✅ OpenClaw智能体接入
- ✅ 飞书Bot通知
- ✅ 状态持久化

### v1.0（历史版本）
- 离线回测功能
- 三市场因子策略
- 静态报告生成
