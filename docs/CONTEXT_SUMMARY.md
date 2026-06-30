# 项目上下文总结（截至 CN800 v5 聚宽模拟盘观察期）

> 目的：压缩对话上下文，新会话先读本文件 + `README.md` + `docs/LLM_CONTEXT.md` 快速恢复状态。更新时间：2026-06-30。
> ⚠️ 本文件仅在用户**明确要求**时更新，平时不动。本次更新用于统一 CN800 v5 当前主线。

## 当前事实覆盖层（2026-06-30）

- 当前 A股聚宽模拟盘主线：`scripts/joinquant_cn800_strategy_v5.py`。
- 当前观察文档：`docs/24_cn800_v5_paper_trading_plan.md`。
- 当前默认动作：让 CN800 v5 在聚宽模拟盘运行至少一个自然月，之后回收交易、持仓、日志和净值做归因。
- v9/v10 现在是历史基线和冷启动经验，不再是当前生产/模拟盘策略。
- 后续 agent 工作时先读 `README.md`、`docs/LLM_CONTEXT.md` 和 `docs/24_cn800_v5_paper_trading_plan.md`，再读旧阶段文档。
- 观察期内不因为短期盈亏改 alpha；优先记录执行差异、现金闲置、整手约束、涨跌停/停牌和滑点。

## 用户偏好与项目原则

- 用户 Python/深度学习基础好，量化从零开始；**解释驱动编码**：先讲原理(写 docs/)，再写代码(注释解释"为什么")，再阶段小结。
- 小步前进，每个里程碑结束给总结 + 问下一步方向，**不擅自开新里程碑**。
- 目标：**稳定做出收益**，**最终目标是美股收益最大化**。方法论：信息量优先于模型复杂度(推荐系统式——先丰富特征再加模型)。
- 起步 A股，渐进扩展港股/美股；环境 conda `quant`(Python 3.11)；联网用 `NO_PROXY='*' python scripts/xxx.py`。
- `CLAUDE.md` 要精炼(每字有 token 价值)；README/docs 可详细。
- 借鉴开源：**只借鉴方法论，不抄具体因子**(因子时效性差、数据口径不匹配)。

## 环境与数据源

- conda `quant`；依赖 `requirements.txt`。运行 demo 都要 `conda activate quant` + `NO_PROXY='*'`。
  - ⚠️ `conda run` 会**缓冲 stdout**(进度打印不实时，看着像卡死)；跑长任务直接用环境 python `-u`：`/opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python -u scripts/xxx.py`。
- **A股行情**：akshare **新浪** `stock_zh_a_daily`(自动补 sh/sz 前缀)。⚠️ 不用东财 `stock_zh_a_hist`(拒绝 Python 客户端)。
- **A股基本面**：东财 `stock_value_em`(**实测可用**)，**日频**估值/股本——PE/PB/PS/市值/流通股本。日频特性省掉季报防前视的麻烦。
- **美股行情**：akshare `stock_us_daily`(数十年 qfq，**实测可用**)。⚠️ 接口无 amount 列→us_loader 合成 `amount≈close×volume`。
- **美股基本面**：⚠️ `stock_us_valuation_baidu` 日频估值**已失效**(空响应)。改用季报 `stock_financial_us_analysis_indicator_em(indicator='单季报')`(**M17 实测可用**，回溯2000年/含公告日 NOTICE_DATE/ROE/毛利率/营收净利同比/单季EPS)拼**点状面板**，价值因子靠 TTM_EPS/价格 自算。⚠️**数据陷阱(M17，4只票验证)**：东财把"去年同期对比行"打上本次公告日→老季度 NOTICE_DATE 系统性晚标约1年(gap≈398天 vs 正常~34天)，不修不会未来函数但基本面滞后近一年劣化回测；修正 `effective_notice=min(原始, REPORT_DATE+60天)`(10-Q法定窗口内绝不早于真实披露)。
- 历史窗口：`config.HISTORY_START=20180101`(M14扩到2018覆盖多regime)。⚠️缓存优先读本地，扩历史须先跑 `scripts/refetch_history.py` 强制覆盖重拉。
- 本地缓存：A股 `data/raw/{symbol}.parquet`(行情)+`{symbol}_value.parquet`(估值)；美股 `data/raw/us/{symbol}.parquet`(行情)+`{symbol}_fund.parquet`(季报基本面，M17)，物理隔离。git 忽略。
- 股票池：A股 `DEFAULT_POOL` 92只去重后89只可用(3只 600837/601989/002013 接口空，自动跳过)；美股 `US_POOL` ~80只跨行业大盘(81只可用)。

## 当前代码结构

```text
quant/
├── config.py                      # 路径常量 + HISTORY_START/END
├── data/
│   ├── akshare_loader.py          # A股行情获取+parquet
│   ├── us_loader.py               # 美股行情(stock_us_daily)，无amount→合成，缓存data/raw/us/
│   ├── fundamental_loader.py      # A股日频估值/股本(stock_value_em)，负PE置NaN
│   ├── us_fundamental_loader.py   # 美股季报基本面(单季报)，TTM EPS，公告日修正min(原始,report+60)，缓存us/{sym}_fund.parquet (M17)
│   ├── universe.py                # A股 DEFAULT_POOL ~90只 / SMALL_POOL 10只
│   ├── universe_us.py             # 美股 US_POOL ~80只跨行业大盘
│   ├── industry.py                # A股行业映射(手工内置17行业，中性化用)
│   └── panel.py                   # build_ohlcv_panels(loader=)/build_value_panels/build_us_fundamental_panels(对齐键=公告日防前视,M17)
├── factor/
│   ├── factors.py                 # 价格/量价+基本面(A股:价值/质量ROE=PB÷PE/成长1÷PEG/现金流1÷PCF;美股M17:us_earnings_yield=TTM_EPS÷价格/us_quality_roe/us_growth)+combine_factors
│   ├── neutralize.py              # 行业去均值/市值回归残差/双中性
│   └── composite.py               # 多因子合成(factor_correlation/ic_weighted_composite/weighted_composite[ICIR×多切分])
├── strategy/
│   ├── dual_ma.py                 # 双均线，signal shift(1)
│   └── mean_reversion.py          # z-score均值回归，stop_loss可选
└── backtest/
    ├── engine.py metrics.py param_scan.py split.py batch.py
    ├── us_cost.py                 # 美股费用模型(每股费+每笔最低费)→make_layered_cost_fn cost_fn回调 (M17)
    ├── portfolio.py               # 横截面组合top-N/等权/再平衡/换手成本
    ├── portfolio_validation.py    # train_test_validate/rolling_walk_forward/stability_summary
    ├── ic_analysis.py             # forward_returns/daily_ic(Spearman)/ic_summary/cumulative_ic
    ├── factor_validation.py       # ic_train_test(裁horizon重叠)/orient_by_ic/build_oriented_composite
    └── layered.py                 # layered_backtest/layer_summary + long_top_layer(分层多头L5)，加cost_fn=注入点默认None向后兼容(M17)
```

## 里程碑结论(M1~M12 精简，详见 docs/ 与 git)

- **M1~M4**：数据管道、单票双均线/均值回归回测、样本内外切分、批量+止损。结论：**单票择时普遍跑不赢买入持有**。
- **M5~M6 横截面组合**：M5 样本内略胜基准，但 M6 滚动验证**未通过样本外**(含选择偏差)。
- **M7~M8 因子IC/分层+扩池**：扩到~90股+量价因子库。全样本下波动因子 IC 看似显著。
- **M9 因子样本外验证(关键)**：IC 按日切 train/test，**量价因子 OOS IC 普遍变号**(过拟合/regime切换非真alpha)。防作弊：单因子正反向都作候选、只在 train 段选方向。
- **M10 基本面因子**：接 `stock_value_em`，加价值/小市值/真实换手率因子。**价值因子 OOS IC 同号显著，比量价稳**；但组合仍跑不赢普涨。
- **M12 行业/市值中性化(关键诊断)**：诊断出**跑输是策略/组合问题非选股问题**(价值top10有9只银行+地产押行业beta)；行业中性化把 earnings_yield ICIR 0.30→0.39、OOS test t 2.42→7.33。nuance：中性化提升【统计稳定性】但不保证某段【组合夏普】更高。A股做空约束→只能分层多头(多空L5−L1夏普≈0)。

## 里程碑结论(M13~M16 详细，本轮重点)

- **M13 多因子IC加权合成(诚实负结果)**：`composite.py`(权重只用切分日前IC防前视，相关>0.7去冗余)。基线 |IC| 单点 + 打磨版 ICIR降噪×多切分点平均带符号权重。**两个池子 ICIR×多切分都稳优于|IC|单点**，但**全池合成 IC 仍负**：5因子里4个在70/30切分处 train→test 整体变号(2025Q3 regime断层)，多切分压不住整体断层；真正 OOS 同号只 earnings_yield 一个。**根因=信息量不足非加权方法**(价值占3个冗余)。教训：加权无法凭空造信号，下一步转向"补正交因子+扩历史"。

- **M14 长历史+正交因子+等权合成(关键突破，项目首个稳定多因子超额)**：对症 M13 两步走——①扩史到2018(`refetch_history.py`+HISTORY_START，regime覆盖 4→23 滚动窗口)；②零成本正交因子(quality_roe=PB÷PE即ROE，growth=1÷PEG，cashflow_yield=1÷PCF，与价值相关仅0.08~0.41)。**结果**：全池滚动 walk-forward(train480/test120/step60，23窗口)**等权合成跑赢83%/超额夏普+0.37/全样本+251%夏普0.90**，胜过最优单因子 earnings_yield(70%/+0.20)和基准(148%/0.69)、回撤更低。**核心教训:等权>IC加权(反直觉)**——IC加权要用train段IC锁定方向/权重，A股因子方向随regime翻号(quality与value相关−0.46几乎反相位)，锁定的旧方向到test押反、反相位因子静态加权下相互抵消(IC加权仅39%)；等权+固定正向先验"不预测方向"绕开此"方向锁定陷阱"。前提：等权只对方向先验稳(全样本IC净正向)的因子有效，`select_stable_positive` 剔除方向漂移的 quality_roe。小池每层股票少噪声大，等权优势在全池才显现(池子规模本身是信息量)。

- **M15 稳健性消融(确认非运气)**：`ablation_demo.py` 单轴消融(固定基线每次只改一个维度：窗口/再平衡/分层)，**8组等权 100% 跑赢过半/100%≥单因子/100%≥IC加权/超额夏普中位+0.33**——83%是结构性优势非超参运气。最干净的"不重叠step"组反而最强(87%)；最弱"分层3"(57%，分层越粗alpha越稀释但仍≥单因子)；IC加权8组7组超额夏普为负再坐实方向锁定陷阱。

- **M16 跨市场美股 Stage 2a(方法论迁移成功，量价负结果)**：**数据层最小改动复用全部上层**——`us_loader.py`+`universe_us.py`+`panel.py`加`loader=`参数注入数据源(默认A股向后兼容，传us_loader即跑美股，**面板及以上零改动**验证分层原则)。**诚实负结果**：美股大盘2018-25量价因子(动量/反转/低波/amihud/趋势/振幅)**无方向稳定净正向者**(select_stable_positive筛出0个)；低波/流动性/振幅IC强但为负(|t|>7=高波动小盘成长跑赢的regime beta非alpha)；6个单因子滚动全跑输基准(最优reversal仅29%/超额−0.14)。**但这恰是方法论迁移成功的证据**：数据层/IC诊断/方向筛选都正常工作且正确拒绝了不稳定因子。**与A股M9跨市场一致**：量价不稳，稳定信号在基本面。

- **M17 美股季报基本面 Stage 2b(预言兑现，项目首个美股稳定跑赢因子)**：对症 M16/M9 结论"稳定信号在基本面"，补上美股基本面因子。**数据源升级**：发现 `stock_financial_us_analysis_indicator_em(indicator='单季报')` 远比预期好(回溯2000年/自带NOTICE_DATE/ROE/毛利率/营收净利同比/单季EPS)。**关键数据陷阱(4只票验证)**：东财把"去年同期对比行"打上本次公告日→老季度NOTICE_DATE系统性晚标约1年(gap≈398天 vs 正常~34天)，不修不会未来函数但基本面滞后近一年；修正 `effective_notice=min(原始,REPORT_DATE+60天)`，修正后严格单调递增无重复。**防前视**：`build_us_fundamental_panels` 对齐键=公告日(非报告期)ffill；TTM EPS 按报告期rolling(4)再绑公告日；脚本设防前视gate人工核对(实测AAPL公告日≤交易日✅)。**美股费用模型** `us_cost.py`(每股费+每笔最低费，区别于A股比例换手；layered_backtest加cost_fn注入点默认None向后兼容)。**结果**：基本面因子**全部meanIC为正**(对比量价0个)，growth_profit滚动跑赢71%/quality_roe 58%(Stage2a量价最优仅29%)；等权合成多头**收益+487%/夏普1.00/回撤29%**，跑赢基准(+344%/0.90/41%)+143pct且回撤低12pct。**诚实nuance**：方向先验随池子构成漂移——小池(30只mega-cap)价值主导(value_ey t+5.99跑赢67%)，全池(~72只)成长接力，select_stable_positive全池只筛出1个→等权未超最优单因子(同M14教训:等权依赖多个稳的正交因子)。8只金融股(BAC/WFC/GS等)接口返回None容错跳过(银行口径特殊)。

## 可运行脚本(关键)

```bash
# A股
NO_PROXY='*' python scripts/factor_validation_demo.py --limit 20 # M9/M10 因子OOS验证+基本面
NO_PROXY='*' python scripts/neutralize_demo.py --limit 20        # M12 中性化消融+分层多头
NO_PROXY='*' python scripts/composite_demo.py --limit 20         # M13 多因子合成(IC加权vs等权)
NO_PROXY='*' python scripts/refetch_history.py                   # M14 扩历史到2018(覆盖旧缓存)
NO_PROXY='*' python scripts/multifactor_demo.py                  # M14 长历史+正交因子+等权合成(83%在这)
NO_PROXY='*' python scripts/ablation_demo.py                     # M15 稳健性消融(8组单轴)
# 美股
NO_PROXY='*' python scripts/us_multifactor_demo.py             # M16 美股量价因子验证(负结果)
NO_PROXY='*' python scripts/us_fundamental_demo.py --limit 30  # M17 美股基本面(小池价值主导)
NO_PROXY='*' python scripts/us_fundamental_demo.py             # M17 美股基本面全池(成长接力，首个稳定跑赢)
# 不带 --limit 即全池。docs/ 编号 01~16 对应讲解。
```

## 关键概念约定(防作弊铁律)

- **防未来函数**：信号必 shift(1)；中性化在单日截面内完成；IC切分裁掉horizon重叠；估值面板 ffill 不 bfill；负PE置NaN。**美股季报基本面须用 NOTICE_DATE(公告日)对齐防前视**。
- 评估只信样本外(test)/滚动；时序不可随机打乱；结论要≥20个跨regime窗口(2年/4窗口不可信)。
- 横截面选股=同一时刻从池里选哪些；组合基准=等权持有全池；回测含手续费+滑点+换手成本。
- A股做空约束→组合靠**分层多头**(L5)，多空L5−L1不可实盘。
- 改策略/接口保持向后兼容(panel 的 loader= 默认A股、stop_loss/first_rebalance 默认值不变)。
- 多因子合成默认**先试等权+固定正向**，只在因子方向确实稳定时才考虑IC加权。

## 下一步：Stage 2c 美股扩池/做空多空对冲(待用户确认是否推进)

**已完成**：M17 美股基本面因子成功——首个美股稳定跑赢的因子(growth_profit滚动71%、等权多头+487%/夏普1.00/回撤29%)，兑现"稳定信号在基本面"的预言。

**M17 留下的核心待解问题**：**方向先验随池子构成漂移**——小池(mega-cap)价值主导、全池成长接力，导致 select_stable_positive 全池只筛出1个稳定因子、等权合成未超最优单因子。这不是方法问题，是池子本身的信息结构。

**用户给的三个美股特性(本轮已用一个)**：①可T+0；②每笔有平台费(M17已建 us_cost 每股费+每笔最低费模型)；③**个股可做多/做空**(A股没有，是关键新能力，本轮按用户选择留到Stage2c)。

**Stage 2c 两个方向(对症方向漂移)**：
1. **扩池纳入中小盘**：中小盘基本面分化更大，价值/成长信号更分明，可能让更多因子方向稳定。需扩 `US_POOL` 并重拉行情+基本面缓存。
2. **做空多空对冲(L5−L1)**：美股可做空，多空对冲能把"方向漂移/市场beta"对冲掉，提取纯alpha。需新写多空回测引擎(参考 layered 的 L5/L1)+借券成本建模。A股因做空约束只能分层多头(多空≈0)，美股这条路是新的。

**成功标准**：扩池后是否有更多方向稳定因子让等权合成超过单因子；或多空对冲后超额夏普是否提升、回撤是否进一步降低。

**路线图剩余**：美股扩池/做空(Stage2c) → 港股复用(`stock_hk_*` 接口，同套代码) → 组合资金管理 → 模拟盘 → 风控。
