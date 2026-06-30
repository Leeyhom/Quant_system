# 量化交易系统 · 工作约定

个人量化学习项目，**循序渐进、解释驱动**：每个模块先讲原理（写进 `docs/`）再写代码，代码注释说明「为什么」。从 A股 起步，渐进扩展港股/美股。

## 环境与运行

- conda 环境 `quant`（Python 3.11）：`conda activate quant`
- 联网拉数据时系统代理会拦截，需绕过：`NO_PROXY='*' python scripts/xxx.py`
- 依赖见 `requirements.txt`

## 数据源

- akshare 的**新浪**接口 `stock_zh_a_daily`（带 sh/sz 前缀，loader 自动补全）。
- ⚠️ 不用默认的 `stock_zh_a_hist`（东方财富，会拒绝 Python 客户端）。
- 基本面：`stock_value_em`（东财，**实测可用**）日频估值/股本——PE/PB/PS/市值/流通股本。日频特性省掉季报防前视的麻烦。
- 历史窗口：`config.HISTORY_START=20180101`（M14扩到2018，覆盖多regime）。行情/估值接口都能回溯到2018-01。⚠️缓存优先读本地，扩历史须先跑 `scripts/refetch_history.py` 强制覆盖重拉。

## 结构

- `quant/config.py` 路径常量+HISTORY_START/END · `quant/data/`（akshare_loader 行情, fundamental_loader 日频估值, universe, industry 行业映射, panel 多字段OHLCV+估值面板）· `quant/factor/`（factors 价格+量价+基本面[价值/质量ROE=PB÷PE/成长1÷PEG/现金流1÷PCF]因子, neutralize 行业/市值中性化, composite 多因子IC加权合成）· `quant/strategy/`（dual_ma, mean_reversion 含可选 stop_loss）· `quant/backtest/`（engine, metrics, param_scan, split, batch, portfolio, portfolio_validation, ic_analysis, factor_validation, layered 含分层多头long_top_layer）
- `scripts/` 可运行 demo（含 refetch_history 扩史 / multifactor_demo 等权合成 / joinquant_cn800_strategy_v5 当前聚宽模拟盘策略）· `docs/` 编号讲解 01~24 · `data/raw/` 本地行情+图(git忽略)
- 分层铁律：数据/因子/策略/回测分离。单票策略输出信号(0/1)；组合策略输出权重。

## 关键约定

- **防未来函数**：信号必须 `shift(1)`（今算明用），否则回测虚高。
- 回测须含手续费+滑点，并与「买入持有」基准对比才有意义。
- 改策略保持向后兼容（如 stop_loss 默认 None=关闭，行为不变）。
- 评估策略要看样本外(test)，不能只看样本内调出来的最优参数。

## 当前主线

- 当前聚宽模拟盘主线：`scripts/joinquant_cn800_strategy_v5.py`（CN800 v5）。
- 观察计划：`docs/24_cn800_v5_paper_trading_plan.md`。
- 现在的默认动作是等待一个月模拟盘结果回收并做归因，不默认继续开新 alpha 版本。
- v9/v10 是 CN800 之前的历史基线和冷启动经验；需要比较时再读取 `docs/22_platform_backtest_and_v9.md` 与 `docs/23_v9_2025_cold_start_v10.md`。

## 进度

- ✅ M1 数据管道 · ✅ M2 单票回测 · ✅ M3 参数/均值回归 · ✅ M4 样本外/批量/止损 · ✅ M5 横截面因子组合 · ✅ M6 因子滚动验证 · ✅ M7 因子IC/分层 · ✅ M8 扩池+量价因子库 · ✅ M9 因子样本外验证+合成 · ✅ M10 基本面因子 · ✅ M12 行业/市值中性化 · ✅ M13 多因子IC加权合成 · ✅ M14 长历史+正交因子+等权合成 · ✅ M15 稳健性消融确认 · ✅ M16 跨市场美股Stage2a · ✅ M17 美股季报基本面Stage2b
- ✅ M1~M19 全部完成 → ⏳ 模拟盘 → 实盘 → 风控（三市场验证全部跑正，详见 README）
- 经验：信息量优先于模型复杂度。**M9**：量价因子全样本显著多为过拟合，OOS IC普遍变号。**M10**：基本面价值因子OOS IC同号显著(比量价稳)，但组合仍跑不赢普涨。**M12诊断(关键)**：跑输是**策略问题非选股问题**——价值因子top10有9只银行+地产(押行业beta)；行业中性化把earnings_yield的ICIR 0.30→0.39、OOS test t 2.42→7.33。**重要nuance**：中性化提升因子【统计稳定性】,但不保证某段样本【组合夏普】更高(放弃了"押对低波行业"的运气换长期可重复性)。A股做空约束→只能分层多头(多空L5-L1夏普≈0)。防作弊:中性化在单日截面内完成无未来函数。**M13(诚实负结果)**：IC加权合成`composite.py`(权重只用切分日前IC,防前视;相关>0.7去冗余)。基线`ic_weighted_composite`(|IC|单点)+打磨`weighted_composite`(ICIR降噪+多切分点平均带符号权重,方向不稳因子被相消收缩)。**两个池子里ICIR×多切分都稳优于|IC|单点(test ICIR limit20 +0.21→+0.32、全池-0.22→-0.16)**,limit20甚至夏普1.91跑赢单因子/基准。**但全池合成IC仍负**:5因子里4个在70/30切分处train→test整体变号(2025Q3 regime断层),多切分只能压切分点间抖动、压不住train/test整体断层;真正OOS同号只earnings_yield一个。**根因=信息量不足非加权方法**:价值占3个(冗余),正交又稳的信息仅1~2份。**教训**:加权是"把已有稳定信号更好组合",无法凭空造信号。下一步从"调加权"转向"补正交因子"(质量/成长/预期修正,与价值低相关)+扩池拉长历史,回到第一性原理"信息量优先于模型复杂度"。**M14(关键突破)**:对症M13诊断两步走——①扩史到2018(`refetch_history.py`+`HISTORY_START`,regime覆盖4→23滚动窗口);②零成本正交因子(quality_roe=PB÷PE即ROE,growth=1÷PEG,cashflow_yield=1÷PCF,与价值相关仅0.08~0.41)。**结果:项目首个稳定多因子超额**——全池滚动walk-forward(train480/test120/step60,23窗口)等权合成跑赢83%/超额夏普+0.37/全样本收益+251%夏普0.90,明显胜过最优单因子earnings_yield(70%/+0.20)和基准(148%/0.69),回撤更低。**核心教训:等权>IC加权(反直觉)**——IC加权要用train段IC锁定方向/权重,A股因子方向随regime翻号(quality与value相关-0.46几乎反相位),锁定的旧方向到test押反、反相位因子静态加权下相互抵消(IC加权仅39%);等权+固定正向先验"不预测方向"绕开此陷阱。**前提**:等权只对方向先验稳(全样本IC净正向)的因子有效——quality_roe虽正交但posRate50%方向漂移,`select_stable_positive`剔除它。小池(40只)每层股票少分层噪声大,等权优势在全池才稳定显现(池子规模本身是信息量)。详见[[m14-equalweight-beats-icweight]]、docs/14。**M15(消融确认)**:`ablation_demo.py`单轴消融(固定基线每次只改一个维度:窗口/再平衡/分层),8组等权100%跑赢过半/100%≥单因子/100%≥IC加权/超额夏普中位+0.33——**83%是结构性优势非超参运气**。最干净的"不重叠step"组反而最强(87%);最弱"分层3"(57%,分层越粗alpha越稀释,但仍≥单因子);IC加权8组7组超额夏普为负再坐实方向锁定陷阱。地基已确认,下一步按用户计划:接港股/美股同步验证(**最终目标美股稳定收益最大化**)→再走组合资金管理。**M16(跨市场美股Stage2a)**:数据层最小改动复用全部上层——`us_loader.py`(stock_us_daily数十年qfq,无amount列→合成close×volume供amihud;缓存独立data/raw/us/),`universe_us.py`(~80只大盘),`panel.py`加`loader=`参数注入数据源(默认A股向后兼容,传us_loader即跑美股,面板及以上零改动验证分层原则)。**诚实负结果**:美股大盘2018-25量价因子无方向稳定净正向者(select_stable_positive筛出0个),低波/流动性/振幅IC强但为负(|t|>7=高波动小盘成长跑赢的regime beta非alpha),6个单因子滚动全跑输基准(最优reversal仅29%/超额-0.14)。**但这恰是方法论迁移成功的证据**:数据层/IC诊断/方向筛选都正常工作且正确拒绝了不稳定因子(没被强负因子骗去反向押注)。**与A股M9跨市场一致**:量价不稳,稳定信号在基本面。下一步Stage2b补季报基本面(NOTICE_DATE防前视)。详见docs/15。⚠️美股stock_us_valuation_baidu日频估值接口已失效(空响应),基本面只能从季报拼点状面板。**M17(美股季报基本面Stage2b,预言兑现)**:对症M16结论"稳定信号在基本面",补上美股基本面因子。**数据源升级**:发现`stock_financial_us_analysis_indicator_em(indicator='单季报')`远比预期好——回溯2000年/自带NOTICE_DATE/含ROE/毛利率/营收净利同比/单季EPS。新增`us_fundamental_loader.py`+`panel.build_us_fundamental_panels`(对齐键=公告日防前视)+factors美股基本面因子(us_earnings_yield=TTM_EPS/价格,us_quality_roe,us_growth)。**关键数据质量陷阱(4只票验证)**:东财把"去年同期对比行"打上本次公告日→老季度NOTICE_DATE系统性晚标约1年(gap≈398天),不修不会未来函数但基本面滞后近一年。修正`effective_notice=min(原始,report+60天)`(10-Q法定窗口内,绝不早于真实披露),修正后公告日严格单调递增无重复。**美股专属费用模型**`us_cost.py`(每股费+每笔最低费,区别于A股比例换手;layered_backtest加cost_fn注入点默认None向后兼容)。**结果=项目首个美股稳定跑赢的因子**:基本面因子全部meanIC为正(对比量价0个稳定者),growth_profit滚动跑赢71%/quality_roe 58%(Stage2a量价最优仅29%);等权合成多头收益+487%夏普1.00回撤29%,跑赢基准+143pct且回撤低12pct。**诚实nuance**:方向先验随池子构成漂移——小池(30只mega-cap)价值主导(value_ey t+5.99跑赢67%),全池成长接力,select_stable_positive全池只筛出1个→等权未超最优单因子(同M14教训:等权依赖多个稳的正交因子)。下一步Stage2c:扩池中小盘(基本面分化更大)+用做空多空对冲(用户指出的美股关键新能力,把方向漂移对冲掉)。详见docs/16、[[m17-us-fundamental-stage2b]]。
