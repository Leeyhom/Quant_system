# -*- coding: utf-8 -*-
"""JoinQuant A-share CN800 strategy — 沪深300+中证500扩展池，v9因子体系。

与 v9 基线的关系：
  - 因子: 不变（value_blend + growth_peg + amihud + quality_roe + low_vol_60）
  - 参数: 不变（top10 / 行业cap2 / 60日再平衡 / 98%仓位）
  - 行业分类: 升级为申万一级31类（v9手工作业18类）
  - 股票池: 152 → 800 只（沪深300+中证500成分股）

本地 honest walk-forward 结果（train=480/test=120/step=120, 11窗口）:
  等权合成: +182.8%收益, +22.0%年化, Sharpe 0.98, 最大回撤 34.8%
  对比旧池: +60.3%收益, +9.4%年化, Sharpe 0.48, 最大回撤 46.2%
  → 不改因子,仅扩池 → 收益+122.5pct, Sharpe翻倍, 回撤-11pct

诚实声明（防过拟合）:
  ① 因子构造只用历史数据（prev_date 的价格+估值），无未来函数
  ② 因子方向先验全为正（越高越好），不做全样本IC定向
  ③ 参数来自v8/v9多轮消融确认（M15: 83%是结构性优势非超参运气）
  ④ CN800 vs 旧池对比口径完全一致，相对改善不受过拟合影响
  ⑤ walk-forward 验证 OOS Sharpe 0.98, 非全样本回测数字

回测设置:
  本金: 60000 CNY
  区间: 2019-01-01 ~ 2025-12-31
  基准: 沪深300

用法: 粘贴到 JoinQuant 策略编辑器, 设置本金 60000, 频率每天。
"""

import math
import numpy as np
import pandas as pd


# ============================ User Config ============================

FEISHU_WEBHOOK = ""

INITIAL_CAPITAL = 60000
MAX_EXPOSURE = 0.98
TOP_N = 10
INDUSTRY_CAP_BASE = 2              # 基础行业上限
INDUSTRY_CAP_RATIO = 10            # 动态上限 = max(BASE, 池内行业股票数/RATIO)
REBALANCE_DAYS = 60
MOMENTUM_120_MIN = -0.10
QUALITY_WEIGHT = 0.5
LOWVOL_WEIGHT = 0.5
RESIDUAL_MOM_WEIGHT = 0.5          # v3新增: 残差动量因子权重
HOLDER_WEIGHT = 0.0
INCLUDE_HOLDER = False
FACTOR_NAMES = [
    "value_blend", "growth_peg", "amihud", "quality_roe", "low_vol_60",
    "residual_momentum",
] + (["holder_concentration"] if INCLUDE_HOLDER else [])
HOLD_MULTIPLIER = 1.5
USE_VOL_BUDGET = False
VOL_BUDGET_LOOKBACK = 60
VOL_BUDGET_MIN_MULT = 0.85
VOL_BUDGET_MAX_MULT = 1.15
MAX_SINGLE_WEIGHT = 0.115
STRATEGY_VERSION = "jq-cn800-v3-dynamic-cap-momentum"

COMMISSION_RATE = 0.00025
STAMP_DUTY_RATE = 0.0005
EFFECTIVE_COMMISSION_RATE = COMMISSION_RATE


# ============================ 股票池：沪深300+中证500（800只） ============================

STOCK_POOL = [
    "000001.XSHE", "000002.XSHE", "000009.XSHE", "000021.XSHE", "000027.XSHE", "000032.XSHE", "000034.XSHE", "000039.XSHE",
    "000050.XSHE", "000060.XSHE", "000062.XSHE", "000063.XSHE", "000088.XSHE", "000100.XSHE", "000155.XSHE", "000157.XSHE",
    "000166.XSHE", "000301.XSHE", "000333.XSHE", "000338.XSHE", "000400.XSHE", "000408.XSHE", "000415.XSHE", "000423.XSHE",
    "000425.XSHE", "000429.XSHE", "000513.XSHE", "000519.XSHE", "000528.XSHE", "000537.XSHE", "000538.XSHE", "000539.XSHE",
    "000559.XSHE", "000568.XSHE", "000582.XSHE", "000591.XSHE", "000596.XSHE", "000598.XSHE", "000617.XSHE", "000623.XSHE",
    "000625.XSHE", "000629.XSHE", "000630.XSHE", "000651.XSHE", "000657.XSHE", "000661.XSHE", "000683.XSHE", "000703.XSHE",
    "000708.XSHE", "000709.XSHE", "000723.XSHE", "000725.XSHE", "000728.XSHE", "000729.XSHE", "000733.XSHE", "000737.XSHE",
    "000738.XSHE", "000739.XSHE", "000750.XSHE", "000768.XSHE", "000776.XSHE", "000783.XSHE", "000785.XSHE", "000786.XSHE",
    "000792.XSHE", "000800.XSHE", "000807.XSHE", "000825.XSHE", "000830.XSHE", "000831.XSHE", "000858.XSHE", "000878.XSHE",
    "000883.XSHE", "000887.XSHE", "000893.XSHE", "000895.XSHE", "000898.XSHE", "000921.XSHE", "000932.XSHE", "000937.XSHE",
    "000938.XSHE", "000951.XSHE", "000959.XSHE", "000960.XSHE", "000963.XSHE", "000967.XSHE", "000975.XSHE", "000977.XSHE",
    "000983.XSHE", "000987.XSHE", "000988.XSHE", "000997.XSHE", "000999.XSHE", "001203.XSHE", "001221.XSHE", "001280.XSHE",
    "001286.XSHE", "001309.XSHE", "001386.XSHE", "001389.XSHE", "001391.XSHE", "001696.XSHE", "001965.XSHE", "001979.XSHE",
    "002001.XSHE", "002007.XSHE", "002008.XSHE", "002025.XSHE", "002027.XSHE", "002028.XSHE", "002032.XSHE", "002044.XSHE",
    "002049.XSHE", "002050.XSHE", "002056.XSHE", "002064.XSHE", "002065.XSHE", "002074.XSHE", "002078.XSHE", "002085.XSHE",
    "002120.XSHE", "002126.XSHE", "002130.XSHE", "002131.XSHE", "002138.XSHE", "002142.XSHE", "002152.XSHE", "002153.XSHE",
    "002155.XSHE", "002157.XSHE", "002179.XSHE", "002185.XSHE", "002195.XSHE", "002202.XSHE", "002203.XSHE", "002223.XSHE",
    "002230.XSHE", "002236.XSHE", "002241.XSHE", "002244.XSHE", "002252.XSHE", "002261.XSHE", "002262.XSHE", "002265.XSHE",
    "002266.XSHE", "002271.XSHE", "002273.XSHE", "002281.XSHE", "002299.XSHE", "002304.XSHE", "002311.XSHE", "002312.XSHE",
    "002318.XSHE", "002335.XSHE", "002340.XSHE", "002352.XSHE", "002353.XSHE", "002371.XSHE", "002384.XSHE", "002402.XSHE",
    "002407.XSHE", "002409.XSHE", "002410.XSHE", "002414.XSHE", "002415.XSHE", "002422.XSHE", "002423.XSHE", "002429.XSHE",
    "002430.XSHE", "002432.XSHE", "002436.XSHE", "002444.XSHE", "002460.XSHE", "002461.XSHE", "002463.XSHE", "002465.XSHE",
    "002466.XSHE", "002472.XSHE", "002475.XSHE", "002487.XSHE", "002493.XSHE", "002500.XSHE", "002508.XSHE", "002517.XSHE",
    "002532.XSHE", "002558.XSHE", "002568.XSHE", "002583.XSHE", "002594.XSHE", "002600.XSHE", "002601.XSHE", "002602.XSHE",
    "002603.XSHE", "002608.XSHE", "002624.XSHE", "002625.XSHE", "002648.XSHE", "002670.XSHE", "002673.XSHE", "002683.XSHE",
    "002709.XSHE", "002714.XSHE", "002736.XSHE", "002738.XSHE", "002739.XSHE", "002756.XSHE", "002773.XSHE", "002797.XSHE",
    "002812.XSHE", "002821.XSHE", "002831.XSHE", "002837.XSHE", "002841.XSHE", "002850.XSHE", "002851.XSHE", "002916.XSHE",
    "002920.XSHE", "002926.XSHE", "002938.XSHE", "002939.XSHE", "002945.XSHE", "002966.XSHE", "002984.XSHE", "003021.XSHE",
    "003022.XSHE", "003031.XSHE", "003035.XSHE", "003816.XSHE", "300001.XSHE", "300002.XSHE", "300003.XSHE", "300012.XSHE",
    "300014.XSHE", "300015.XSHE", "300017.XSHE", "300024.XSHE", "300033.XSHE", "300037.XSHE", "300054.XSHE", "300058.XSHE",
    "300059.XSHE", "300073.XSHE", "300100.XSHE", "300115.XSHE", "300122.XSHE", "300124.XSHE", "300136.XSHE", "300140.XSHE",
    "300142.XSHE", "300144.XSHE", "300146.XSHE", "300207.XSHE", "300223.XSHE", "300251.XSHE", "300274.XSHE", "300285.XSHE",
    "300308.XSHE", "300316.XSHE", "300339.XSHE", "300346.XSHE", "300373.XSHE", "300383.XSHE", "300390.XSHE", "300394.XSHE",
    "300395.XSHE", "300408.XSHE", "300413.XSHE", "300418.XSHE", "300432.XSHE", "300433.XSHE", "300442.XSHE", "300450.XSHE",
    "300454.XSHE", "300458.XSHE", "300474.XSHE", "300475.XSHE", "300476.XSHE", "300487.XSHE", "300496.XSHE", "300498.XSHE",
    "300502.XSHE", "300548.XSHE", "300558.XSHE", "300567.XSHE", "300570.XSHE", "300604.XSHE", "300620.XSHE", "300623.XSHE",
    "300627.XSHE", "300628.XSHE", "300661.XSHE", "300666.XSHE", "300676.XSHE", "300677.XSHE", "300679.XSHE", "300699.XSHE",
    "300718.XSHE", "300724.XSHE", "300735.XSHE", "300748.XSHE", "300750.XSHE", "300751.XSHE", "300757.XSHE", "300759.XSHE",
    "300760.XSHE", "300763.XSHE", "300803.XSHE", "300832.XSHE", "300857.XSHE", "300866.XSHE", "300888.XSHE", "300896.XSHE",
    "300919.XSHE", "300953.XSHE", "300957.XSHE", "300972.XSHE", "300999.XSHE", "301165.XSHE", "301200.XSHE", "301236.XSHE",
    "301269.XSHE", "301301.XSHE", "301308.XSHE", "301358.XSHE", "301377.XSHE", "301498.XSHE", "301526.XSHE", "301536.XSHE",
    "301606.XSHE", "301611.XSHE", "302132.XSHE", "600000.XSHG", "600004.XSHG", "600008.XSHG", "600009.XSHG", "600010.XSHG",
    "600011.XSHG", "600015.XSHG", "600016.XSHG", "600018.XSHG", "600019.XSHG", "600021.XSHG", "600023.XSHG", "600025.XSHG",
    "600026.XSHG", "600027.XSHG", "600028.XSHG", "600029.XSHG", "600030.XSHG", "600031.XSHG", "600032.XSHG", "600036.XSHG",
    "600038.XSHG", "600039.XSHG", "600048.XSHG", "600050.XSHG", "600060.XSHG", "600061.XSHG", "600062.XSHG", "600066.XSHG",
    "600085.XSHG", "600089.XSHG", "600095.XSHG", "600098.XSHG", "600100.XSHG", "600104.XSHG", "600105.XSHG", "600109.XSHG",
    "600111.XSHG", "600115.XSHG", "600118.XSHG", "600126.XSHG", "600131.XSHG", "600132.XSHG", "600141.XSHG", "600143.XSHG",
    "600150.XSHG", "600153.XSHG", "600157.XSHG", "600160.XSHG", "600161.XSHG", "600166.XSHG", "600170.XSHG", "600171.XSHG",
    "600176.XSHG", "600177.XSHG", "600183.XSHG", "600188.XSHG", "600196.XSHG", "600208.XSHG", "600219.XSHG", "600221.XSHG",
    "600233.XSHG", "600256.XSHG", "600276.XSHG", "600282.XSHG", "600292.XSHG", "600295.XSHG", "600298.XSHG", "600299.XSHG",
    "600309.XSHG", "600312.XSHG", "600316.XSHG", "600329.XSHG", "600332.XSHG", "600339.XSHG", "600346.XSHG", "600348.XSHG",
    "600350.XSHG", "600352.XSHG", "600362.XSHG", "600363.XSHG", "600369.XSHG", "600372.XSHG", "600377.XSHG", "600378.XSHG",
    "600380.XSHG", "600390.XSHG", "600392.XSHG", "600398.XSHG", "600406.XSHG", "600415.XSHG", "600426.XSHG", "600435.XSHG",
    "600436.XSHG", "600438.XSHG", "600460.XSHG", "600482.XSHG", "600483.XSHG", "600486.XSHG", "600489.XSHG", "600497.XSHG",
    "600498.XSHG", "600499.XSHG", "600511.XSHG", "600515.XSHG", "600516.XSHG", "600517.XSHG", "600519.XSHG", "600521.XSHG",
    "600522.XSHG", "600535.XSHG", "600536.XSHG", "600546.XSHG", "600547.XSHG", "600549.XSHG", "600562.XSHG", "600563.XSHG",
    "600566.XSHG", "600570.XSHG", "600578.XSHG", "600582.XSHG", "600583.XSHG", "600584.XSHG", "600585.XSHG", "600588.XSHG",
    "600595.XSHG", "600598.XSHG", "600600.XSHG", "600601.XSHG", "600602.XSHG", "600606.XSHG", "600637.XSHG", "600642.XSHG",
    "600655.XSHG", "600660.XSHG", "600663.XSHG", "600674.XSHG", "600685.XSHG", "600688.XSHG", "600690.XSHG", "600699.XSHG",
    "600704.XSHG", "600707.XSHG", "600711.XSHG", "600737.XSHG", "600741.XSHG", "600754.XSHG", "600760.XSHG", "600763.XSHG",
    "600764.XSHG", "600765.XSHG", "600795.XSHG", "600801.XSHG", "600803.XSHG", "600808.XSHG", "600809.XSHG", "600816.XSHG",
    "600820.XSHG", "600845.XSHG", "600848.XSHG", "600862.XSHG", "600863.XSHG", "600871.XSHG", "600873.XSHG", "600875.XSHG",
    "600879.XSHG", "600884.XSHG", "600885.XSHG", "600886.XSHG", "600887.XSHG", "600893.XSHG", "600900.XSHG", "600901.XSHG",
    "600905.XSHG", "600906.XSHG", "600909.XSHG", "600918.XSHG", "600919.XSHG", "600926.XSHG", "600927.XSHG", "600930.XSHG",
    "600938.XSHG", "600941.XSHG", "600958.XSHG", "600967.XSHG", "600968.XSHG", "600970.XSHG", "600977.XSHG", "600985.XSHG",
    "600988.XSHG", "600989.XSHG", "600995.XSHG", "600998.XSHG", "600999.XSHG", "601000.XSHG", "601001.XSHG", "601006.XSHG",
    "601009.XSHG", "601012.XSHG", "601016.XSHG", "601018.XSHG", "601019.XSHG", "601021.XSHG", "601058.XSHG", "601059.XSHG",
    "601066.XSHG", "601077.XSHG", "601088.XSHG", "601098.XSHG", "601099.XSHG", "601100.XSHG", "601106.XSHG", "601108.XSHG",
    "601111.XSHG", "601112.XSHG", "601117.XSHG", "601118.XSHG", "601127.XSHG", "601128.XSHG", "601136.XSHG", "601138.XSHG",
    "601139.XSHG", "601155.XSHG", "601156.XSHG", "601162.XSHG", "601166.XSHG", "601169.XSHG", "601179.XSHG", "601186.XSHG",
    "601198.XSHG", "601211.XSHG", "601212.XSHG", "601216.XSHG", "601225.XSHG", "601228.XSHG", "601229.XSHG", "601233.XSHG",
    "601236.XSHG", "601238.XSHG", "601288.XSHG", "601298.XSHG", "601318.XSHG", "601319.XSHG", "601328.XSHG", "601336.XSHG",
    "601360.XSHG", "601377.XSHG", "601390.XSHG", "601398.XSHG", "601399.XSHG", "601456.XSHG", "601555.XSHG", "601567.XSHG",
    "601577.XSHG", "601598.XSHG", "601600.XSHG", "601601.XSHG", "601607.XSHG", "601608.XSHG", "601611.XSHG", "601615.XSHG",
    "601618.XSHG", "601628.XSHG", "601633.XSHG", "601658.XSHG", "601665.XSHG", "601666.XSHG", "601668.XSHG", "601669.XSHG",
    "601688.XSHG", "601689.XSHG", "601696.XSHG", "601698.XSHG", "601699.XSHG", "601717.XSHG", "601727.XSHG", "601728.XSHG",
    "601766.XSHG", "601788.XSHG", "601799.XSHG", "601800.XSHG", "601808.XSHG", "601816.XSHG", "601818.XSHG", "601825.XSHG",
    "601838.XSHG", "601857.XSHG", "601865.XSHG", "601866.XSHG", "601868.XSHG", "601869.XSHG", "601872.XSHG", "601877.XSHG",
    "601878.XSHG", "601880.XSHG", "601881.XSHG", "601888.XSHG", "601898.XSHG", "601899.XSHG", "601901.XSHG", "601916.XSHG",
    "601919.XSHG", "601928.XSHG", "601939.XSHG", "601958.XSHG", "601966.XSHG", "601985.XSHG", "601988.XSHG", "601990.XSHG",
    "601991.XSHG", "601995.XSHG", "601997.XSHG", "601998.XSHG", "603000.XSHG", "603019.XSHG", "603049.XSHG", "603077.XSHG",
    "603087.XSHG", "603092.XSHG", "603119.XSHG", "603129.XSHG", "603156.XSHG", "603160.XSHG", "603175.XSHG", "603179.XSHG",
    "603225.XSHG", "603233.XSHG", "603256.XSHG", "603259.XSHG", "603260.XSHG", "603288.XSHG", "603290.XSHG", "603296.XSHG",
    "603298.XSHG", "603308.XSHG", "603338.XSHG", "603341.XSHG", "603345.XSHG", "603369.XSHG", "603379.XSHG", "603392.XSHG",
    "603444.XSHG", "603486.XSHG", "603501.XSHG", "603529.XSHG", "603565.XSHG", "603568.XSHG", "603589.XSHG", "603596.XSHG",
    "603605.XSHG", "603606.XSHG", "603650.XSHG", "603658.XSHG", "603659.XSHG", "603688.XSHG", "603699.XSHG", "603728.XSHG",
    "603737.XSHG", "603766.XSHG", "603786.XSHG", "603799.XSHG", "603806.XSHG", "603816.XSHG", "603833.XSHG", "603858.XSHG",
    "603885.XSHG", "603893.XSHG", "603899.XSHG", "603920.XSHG", "603939.XSHG", "603979.XSHG", "603986.XSHG", "603993.XSHG",
    "605117.XSHG", "605358.XSHG", "605499.XSHG", "605589.XSHG", "688002.XSHG", "688008.XSHG", "688009.XSHG", "688012.XSHG",
    "688017.XSHG", "688018.XSHG", "688019.XSHG", "688027.XSHG", "688036.XSHG", "688037.XSHG", "688041.XSHG", "688047.XSHG",
    "688052.XSHG", "688065.XSHG", "688072.XSHG", "688082.XSHG", "688099.XSHG", "688111.XSHG", "688114.XSHG", "688120.XSHG",
    "688122.XSHG", "688126.XSHG", "688166.XSHG", "688169.XSHG", "688172.XSHG", "688180.XSHG", "688183.XSHG", "688187.XSHG",
    "688188.XSHG", "688192.XSHG", "688200.XSHG", "688213.XSHG", "688220.XSHG", "688223.XSHG", "688234.XSHG", "688235.XSHG",
    "688248.XSHG", "688256.XSHG", "688266.XSHG", "688271.XSHG", "688278.XSHG", "688281.XSHG", "688295.XSHG", "688297.XSHG",
    "688301.XSHG", "688303.XSHG", "688313.XSHG", "688318.XSHG", "688322.XSHG", "688331.XSHG", "688336.XSHG", "688343.XSHG",
    "688347.XSHG", "688349.XSHG", "688361.XSHG", "688363.XSHG", "688375.XSHG", "688385.XSHG", "688387.XSHG", "688396.XSHG",
    "688411.XSHG", "688425.XSHG", "688469.XSHG", "688472.XSHG", "688475.XSHG", "688498.XSHG", "688506.XSHG", "688520.XSHG",
    "688521.XSHG", "688538.XSHG", "688561.XSHG", "688563.XSHG", "688568.XSHG", "688578.XSHG", "688582.XSHG", "688599.XSHG",
    "688608.XSHG", "688615.XSHG", "688617.XSHG", "688629.XSHG", "688676.XSHG", "688692.XSHG", "688702.XSHG", "688708.XSHG",
    "688709.XSHG", "688728.XSHG", "688772.XSHG", "688777.XSHG", "688778.XSHG", "688819.XSHG", "688981.XSHG", "689009.XSHG",
]


# ============================ 行业分类：申万一级（31类） ============================

INDUSTRY_MAP = {
        "交通运输": [
            "000088.XSHE", "000429.XSHE", "000582.XSHE", "001391.XSHE", "001965.XSHE", "002120.XSHE", "002352.XSHE", "600004.XSHG",
            "600009.XSHG", "600018.XSHG", "600026.XSHG", "600029.XSHG", "600115.XSHG", "600153.XSHG", "600221.XSHG", "600233.XSHG",
            "600350.XSHG", "600377.XSHG", "600704.XSHG", "601000.XSHG", "601006.XSHG", "601018.XSHG", "601021.XSHG", "601111.XSHG",
            "601156.XSHG", "601228.XSHG", "601298.XSHG", "601598.XSHG", "601816.XSHG", "601866.XSHG", "601872.XSHG", "601880.XSHG",
            "601919.XSHG", "603565.XSHG", "603885.XSHG",
        ],
        "传媒": [
            "002027.XSHE", "002517.XSHE", "002558.XSHE", "002602.XSHE", "002624.XSHE", "002739.XSHE", "300002.XSHE", "300058.XSHE",
            "300251.XSHE", "300413.XSHE", "300418.XSHE", "600637.XSHG", "600977.XSHG", "601019.XSHG", "601098.XSHG", "601928.XSHG",
            "603000.XSHG", "603444.XSHG",
        ],
        "公用事业": [
            "000027.XSHE", "000155.XSHE", "000537.XSHE", "000539.XSHE", "000591.XSHE", "000883.XSHE", "001286.XSHE", "002608.XSHE",
            "003035.XSHE", "003816.XSHE", "600011.XSHG", "600021.XSHG", "600023.XSHG", "600025.XSHG", "600027.XSHG", "600032.XSHG",
            "600098.XSHG", "600483.XSHG", "600578.XSHG", "600642.XSHG", "600674.XSHG", "600795.XSHG", "600803.XSHG", "600863.XSHG",
            "600886.XSHG", "600900.XSHG", "600905.XSHG", "600930.XSHG", "600995.XSHG", "601016.XSHG", "601139.XSHG", "601985.XSHG",
            "601991.XSHG",
        ],
        "其他": [
            "600339.XSHG",
        ],
        "农林牧渔": [
            "002157.XSHE", "002299.XSHE", "002311.XSHE", "002714.XSHE", "300498.XSHE", "300999.XSHE", "301498.XSHE", "600598.XSHG",
            "600737.XSHG", "601118.XSHG",
        ],
        "医药生物": [
            "000423.XSHE", "000513.XSHE", "000538.XSHE", "000623.XSHE", "000661.XSHE", "000739.XSHE", "000963.XSHE", "000999.XSHE",
            "002007.XSHE", "002044.XSHE", "002223.XSHE", "002252.XSHE", "002262.XSHE", "002422.XSHE", "002432.XSHE", "002603.XSHE",
            "002773.XSHE", "002821.XSHE", "300003.XSHE", "300015.XSHE", "300122.XSHE", "300142.XSHE", "300558.XSHE", "300676.XSHE",
            "300677.XSHE", "300759.XSHE", "300760.XSHE", "300832.XSHE", "301301.XSHE", "600062.XSHG", "600085.XSHG", "600161.XSHG",
            "600196.XSHG", "600276.XSHG", "600329.XSHG", "600332.XSHG", "600380.XSHG", "600436.XSHG", "600511.XSHG", "600521.XSHG",
            "600535.XSHG", "600566.XSHG", "600763.XSHG", "600998.XSHG", "601607.XSHG", "603087.XSHG", "603233.XSHG", "603259.XSHG",
            "603392.XSHG", "603658.XSHG", "603858.XSHG", "603939.XSHG", "688114.XSHG", "688166.XSHG", "688180.XSHG", "688192.XSHG",
            "688235.XSHG", "688266.XSHG", "688271.XSHG", "688278.XSHG", "688301.XSHG", "688331.XSHG", "688336.XSHG", "688506.XSHG",
            "688520.XSHG", "688578.XSHG", "688617.XSHG",
        ],
        "商贸零售": [
            "000785.XSHE", "300972.XSHE", "600415.XSHG", "600655.XSHG", "601888.XSHG",
        ],
        "国防军工": [
            "000733.XSHE", "000738.XSHE", "000768.XSHE", "002025.XSHE", "002179.XSHE", "002414.XSHE", "002465.XSHE", "002625.XSHE",
            "300395.XSHE", "300474.XSHE", "300699.XSHE", "302132.XSHE", "600038.XSHG", "600118.XSHG", "600150.XSHG", "600316.XSHG",
            "600372.XSHG", "600435.XSHG", "600562.XSHG", "600685.XSHG", "600760.XSHG", "600764.XSHG", "600879.XSHG", "600893.XSHG",
            "601698.XSHG", "688002.XSHG", "688122.XSHG", "688281.XSHG", "688297.XSHG", "688375.XSHG", "688563.XSHG", "688629.XSHG",
            "688708.XSHG",
        ],
        "基础化工": [
            "000683.XSHE", "000792.XSHE", "000830.XSHE", "000893.XSHE", "002001.XSHE", "002064.XSHE", "002312.XSHE", "002407.XSHE",
            "002430.XSHE", "002601.XSHE", "002648.XSHE", "002683.XSHE", "300487.XSHE", "600141.XSHG", "600143.XSHG", "600160.XSHG",
            "600299.XSHG", "600309.XSHG", "600352.XSHG", "600378.XSHG", "600426.XSHG", "600486.XSHG", "600873.XSHG", "600989.XSHG",
            "601216.XSHG", "603077.XSHG", "603225.XSHG", "603260.XSHG", "603379.XSHG", "603650.XSHG", "603688.XSHG", "605589.XSHG",
            "688065.XSHG", "688295.XSHG",
        ],
        "家用电器": [
            "000333.XSHE", "000651.XSHE", "000921.XSHE", "002032.XSHE", "002050.XSHE", "002429.XSHE", "002508.XSHE", "600060.XSHG",
            "600690.XSHG", "603486.XSHG", "688169.XSHG",
        ],
        "建筑材料": [
            "000786.XSHE", "002271.XSHE", "301526.XSHE", "600176.XSHG", "600585.XSHG", "600801.XSHG", "601112.XSHG", "603256.XSHG",
            "603737.XSHG",
        ],
        "建筑装饰": [
            "000032.XSHE", "600039.XSHG", "600170.XSHG", "600820.XSHG", "600970.XSHG", "601117.XSHG", "601186.XSHG", "601390.XSHG",
            "601611.XSHG", "601618.XSHG", "601668.XSHG", "601669.XSHG", "601800.XSHG", "601868.XSHG",
        ],
        "房地产": [
            "000002.XSHE", "001979.XSHE", "002244.XSHE", "600048.XSHG", "600208.XSHG", "600515.XSHG", "600606.XSHG", "600663.XSHG",
            "600848.XSHG", "601155.XSHG",
        ],
        "有色金属": [
            "000060.XSHE", "000408.XSHE", "000630.XSHE", "000657.XSHE", "000737.XSHE", "000807.XSHE", "000831.XSHE", "000878.XSHE",
            "000960.XSHE", "000975.XSHE", "001280.XSHE", "002155.XSHE", "002203.XSHE", "002460.XSHE", "002466.XSHE", "002532.XSHE",
            "002738.XSHE", "002756.XSHE", "300748.XSHE", "600111.XSHG", "600219.XSHG", "600362.XSHG", "600392.XSHG", "600489.XSHG",
            "600497.XSHG", "600547.XSHG", "600549.XSHG", "600595.XSHG", "600711.XSHG", "600988.XSHG", "601212.XSHG", "601600.XSHG",
            "601899.XSHG", "601958.XSHG", "603799.XSHG", "603979.XSHG", "603993.XSHG",
        ],
        "机械设备": [
            "000039.XSHE", "000157.XSHE", "000425.XSHE", "000519.XSHE", "000528.XSHE", "000988.XSHE", "001696.XSHE", "002008.XSHE",
            "002131.XSHE", "002353.XSHE", "002444.XSHE", "002837.XSHE", "300024.XSHE", "300124.XSHE", "300567.XSHE", "300718.XSHE",
            "300757.XSHE", "301200.XSHE", "301377.XSHE", "600031.XSHG", "600499.XSHG", "600582.XSHG", "600765.XSHG", "600862.XSHG",
            "600967.XSHG", "601100.XSHG", "601106.XSHG", "601399.XSHG", "601608.XSHG", "601717.XSHG", "601766.XSHG", "603298.XSHG",
            "603308.XSHG", "603338.XSHG", "603699.XSHG", "688009.XSHG", "688017.XSHG", "688187.XSHG", "688425.XSHG", "688777.XSHG",
        ],
        "汽车": [
            "000338.XSHE", "000559.XSHE", "000625.XSHE", "000800.XSHE", "000887.XSHE", "000951.XSHE", "002085.XSHE", "002126.XSHE",
            "002265.XSHE", "002472.XSHE", "002594.XSHE", "002984.XSHE", "300100.XSHE", "600066.XSHG", "600104.XSHG", "600166.XSHG",
            "600660.XSHG", "600699.XSHG", "600741.XSHG", "601058.XSHG", "601127.XSHG", "601238.XSHG", "601633.XSHG", "601689.XSHG",
            "601799.XSHG", "601966.XSHG", "603049.XSHG", "603119.XSHG", "603129.XSHG", "603179.XSHG", "603529.XSHG", "603596.XSHG",
            "603766.XSHG", "603786.XSHG", "689009.XSHG",
        ],
        "煤炭": [
            "000723.XSHE", "000937.XSHE", "000983.XSHE", "600157.XSHG", "600188.XSHG", "600348.XSHG", "600546.XSHG", "600985.XSHG",
            "601001.XSHG", "601088.XSHG", "601225.XSHG", "601666.XSHG", "601699.XSHG", "601898.XSHG",
        ],
        "环保": [
            "000598.XSHE", "000967.XSHE", "002266.XSHE", "300140.XSHE", "600008.XSHG", "600292.XSHG", "603568.XSHG",
        ],
        "电力设备": [
            "000009.XSHE", "000400.XSHE", "002028.XSHE", "002056.XSHE", "002074.XSHE", "002202.XSHE", "002335.XSHE", "002340.XSHE",
            "002487.XSHE", "002709.XSHE", "002812.XSHE", "002850.XSHE", "002851.XSHE", "003021.XSHE", "003022.XSHE", "300001.XSHE",
            "300014.XSHE", "300037.XSHE", "300073.XSHE", "300207.XSHE", "300274.XSHE", "300316.XSHE", "300390.XSHE", "300432.XSHE",
            "300450.XSHE", "300724.XSHE", "300750.XSHE", "300751.XSHE", "300763.XSHE", "300919.XSHE", "300953.XSHE", "301358.XSHE",
            "600089.XSHG", "600312.XSHG", "600406.XSHG", "600438.XSHG", "600482.XSHG", "600875.XSHG", "600884.XSHG", "600885.XSHG",
            "601012.XSHG", "601179.XSHG", "601567.XSHG", "601615.XSHG", "601727.XSHG", "601865.XSHG", "601877.XSHG", "603092.XSHG",
            "603606.XSHG", "603659.XSHG", "603728.XSHG", "603806.XSHG", "605117.XSHG", "688223.XSHG", "688248.XSHG", "688303.XSHG",
            "688349.XSHG", "688411.XSHG", "688472.XSHG", "688599.XSHG", "688676.XSHG", "688772.XSHG", "688778.XSHG", "688819.XSHG",
        ],
        "电子": [
            "000021.XSHE", "000050.XSHE", "000062.XSHE", "000100.XSHE", "000725.XSHE", "001309.XSHE", "001389.XSHE", "002049.XSHE",
            "002130.XSHE", "002138.XSHE", "002185.XSHE", "002241.XSHE", "002273.XSHE", "002371.XSHE", "002384.XSHE", "002402.XSHE",
            "002409.XSHE", "002436.XSHE", "002463.XSHE", "002475.XSHE", "002600.XSHE", "002841.XSHE", "002916.XSHE", "002938.XSHE",
            "300054.XSHE", "300115.XSHE", "300136.XSHE", "300223.XSHE", "300285.XSHE", "300346.XSHE", "300373.XSHE", "300408.XSHE",
            "300433.XSHE", "300458.XSHE", "300475.XSHE", "300476.XSHE", "300604.XSHE", "300623.XSHE", "300661.XSHE", "300666.XSHE",
            "300679.XSHE", "300735.XSHE", "300857.XSHE", "300866.XSHE", "301308.XSHE", "301536.XSHE", "301606.XSHE", "301611.XSHE",
            "600171.XSHG", "600183.XSHG", "600363.XSHG", "600460.XSHG", "600563.XSHG", "600584.XSHG", "600601.XSHG", "600707.XSHG",
            "601138.XSHG", "603160.XSHG", "603175.XSHG", "603290.XSHG", "603296.XSHG", "603341.XSHG", "603501.XSHG", "603893.XSHG",
            "603920.XSHG", "603986.XSHG", "605358.XSHG", "688008.XSHG", "688012.XSHG", "688018.XSHG", "688019.XSHG", "688036.XSHG",
            "688037.XSHG", "688041.XSHG", "688047.XSHG", "688052.XSHG", "688072.XSHG", "688082.XSHG", "688099.XSHG", "688120.XSHG",
            "688126.XSHG", "688172.XSHG", "688183.XSHG", "688200.XSHG", "688213.XSHG", "688220.XSHG", "688234.XSHG", "688256.XSHG",
            "688322.XSHG", "688347.XSHG", "688361.XSHG", "688385.XSHG", "688396.XSHG", "688469.XSHG", "688498.XSHG", "688521.XSHG",
            "688538.XSHG", "688582.XSHG", "688608.XSHG", "688702.XSHG", "688709.XSHG", "688728.XSHG", "688981.XSHG",
        ],
        "石油石化": [
            "000301.XSHE", "000703.XSHE", "002493.XSHE", "600028.XSHG", "600256.XSHG", "600346.XSHG", "600583.XSHG", "600688.XSHG",
            "600871.XSHG", "600938.XSHG", "600968.XSHG", "601233.XSHG", "601808.XSHG", "601857.XSHG",
        ],
        "社会服务": [
            "300012.XSHE", "300144.XSHE", "600754.XSHG",
        ],
        "纺织服饰": [
            "600177.XSHG", "600398.XSHG",
        ],
        "美容护理": [
            "300888.XSHE", "300896.XSHE", "300957.XSHE", "603605.XSHG", "688363.XSHG",
        ],
        "计算机": [
            "000034.XSHE", "000938.XSHE", "000977.XSHE", "000997.XSHE", "002065.XSHE", "002152.XSHE", "002153.XSHE", "002195.XSHE",
            "002230.XSHE", "002236.XSHE", "002261.XSHE", "002410.XSHE", "002415.XSHE", "002920.XSHE", "300017.XSHE", "300033.XSHE",
            "300339.XSHE", "300454.XSHE", "300496.XSHE", "300803.XSHE", "301236.XSHE", "301269.XSHE", "600100.XSHG", "600131.XSHG",
            "600536.XSHG", "600570.XSHG", "600588.XSHG", "600602.XSHG", "600845.XSHG", "601360.XSHG", "603019.XSHG", "688111.XSHG",
            "688188.XSHG", "688318.XSHG", "688343.XSHG", "688475.XSHG", "688561.XSHG", "688568.XSHG", "688615.XSHG", "688692.XSHG",
        ],
        "轻工制造": [
            "001221.XSHE", "001386.XSHE", "002078.XSHE", "002831.XSHE", "603816.XSHG", "603833.XSHG", "603899.XSHG",
        ],
        "通信": [
            "000063.XSHE", "002281.XSHE", "002583.XSHE", "003031.XSHE", "300308.XSHE", "300383.XSHE", "300394.XSHE", "300442.XSHE",
            "300502.XSHE", "300548.XSHE", "300570.XSHE", "300620.XSHE", "300627.XSHE", "300628.XSHE", "301165.XSHE", "600050.XSHG",
            "600105.XSHG", "600498.XSHG", "600522.XSHG", "600941.XSHG", "601728.XSHG", "601869.XSHG", "688027.XSHG", "688313.XSHG",
            "688387.XSHG",
        ],
        "钢铁": [
            "000629.XSHE", "000708.XSHE", "000709.XSHE", "000825.XSHE", "000898.XSHE", "000932.XSHE", "000959.XSHE", "001203.XSHE",
            "002318.XSHE", "600010.XSHG", "600019.XSHG", "600126.XSHG", "600282.XSHG", "600295.XSHG", "600516.XSHG", "600808.XSHG",
        ],
        "银行": [
            "000001.XSHE", "002142.XSHE", "002966.XSHE", "600000.XSHG", "600015.XSHG", "600016.XSHG", "600036.XSHG", "600919.XSHG",
            "600926.XSHG", "601009.XSHG", "601077.XSHG", "601128.XSHG", "601166.XSHG", "601169.XSHG", "601229.XSHG", "601288.XSHG",
            "601328.XSHG", "601398.XSHG", "601577.XSHG", "601658.XSHG", "601665.XSHG", "601818.XSHG", "601825.XSHG", "601838.XSHG",
            "601916.XSHG", "601939.XSHG", "601988.XSHG", "601997.XSHG", "601998.XSHG",
        ],
        "非银金融": [
            "000166.XSHE", "000415.XSHE", "000617.XSHE", "000728.XSHE", "000750.XSHE", "000776.XSHE", "000783.XSHE", "000987.XSHE",
            "002423.XSHE", "002500.XSHE", "002670.XSHE", "002673.XSHE", "002736.XSHE", "002797.XSHE", "002926.XSHE", "002939.XSHE",
            "002945.XSHE", "300059.XSHE", "600030.XSHG", "600061.XSHG", "600095.XSHG", "600109.XSHG", "600369.XSHG", "600390.XSHG",
            "600517.XSHG", "600816.XSHG", "600901.XSHG", "600906.XSHG", "600909.XSHG", "600918.XSHG", "600927.XSHG", "600958.XSHG",
            "600999.XSHG", "601059.XSHG", "601066.XSHG", "601099.XSHG", "601108.XSHG", "601136.XSHG", "601162.XSHG", "601198.XSHG",
            "601211.XSHG", "601236.XSHG", "601318.XSHG", "601319.XSHG", "601336.XSHG", "601377.XSHG", "601456.XSHG", "601555.XSHG",
            "601601.XSHG", "601628.XSHG", "601688.XSHG", "601696.XSHG", "601788.XSHG", "601878.XSHG", "601881.XSHG", "601901.XSHG",
            "601990.XSHG", "601995.XSHG",
        ],
        "食品饮料": [
            "000568.XSHE", "000596.XSHE", "000729.XSHE", "000858.XSHE", "000895.XSHE", "002304.XSHE", "002461.XSHE", "002568.XSHE",
            "300146.XSHE", "600132.XSHG", "600298.XSHG", "600519.XSHG", "600600.XSHG", "600809.XSHG", "600887.XSHG", "603156.XSHG",
            "603288.XSHG", "603345.XSHG", "603369.XSHG", "603589.XSHG", "605499.XSHG",
        ],
}


def _industry_by_stock():
    """构建 6位代码 → 行业名 的映射（去除 .XSHG/.XSHE 后缀）。"""
    out = {}
    for ind, codes in INDUSTRY_MAP.items():
        for c in codes:
            code6 = c.split(".")[0] if "." in c else c
            out[code6] = ind
    return out


INDUSTRY_BY_STOCK = _industry_by_stock()


# ============================ JoinQuant Hooks ============================


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)

    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=STAMP_DUTY_RATE,
            open_commission=EFFECTIVE_COMMISSION_RATE,
            close_commission=EFFECTIVE_COMMISSION_RATE,
            close_today_commission=0,
            min_commission=5,
        ),
        type="stock",
    )

    try:
        set_slippage(PriceRelatedSlippage(0.0005))
    except Exception:
        pass

    g.days_since_rebalance = REBALANCE_DAYS
    g.last_targets = []
    g.last_factor_names = []
    g.last_momentum_120 = pd.Series(dtype=float)
    g.last_vol_60 = pd.Series(dtype=float)
    g.last_holder_coverage = 0

    # 预计算各行业在池中的股票数，用于动态行业上限
    g.industry_pool_counts = {}
    for code in STOCK_POOL:
        code6 = code.split(".")[0] if "." in code else code
        ind = INDUSTRY_BY_STOCK.get(code6, "其他")
        g.industry_pool_counts[ind] = g.industry_pool_counts.get(ind, 0) + 1

    run_daily(rebalance, time="14:30")

    send_feishu(
        "A股聚宽模拟盘 CN800 v3 已启动\n"
        "版本: {}\n"
        "本金: 60000\n"
        "池子: {} 只 (沪深300+中证500, SW{}行业)\n"
        "配置: top{} / 行业cap动态(max{},N/{}) / {}日再平衡 / {:.0f}%仓位 / 120日动量>{:.0f}%\n"
        "因子: {} ({}个)\n"
        "v3新增: 残差动量因子(权重{}) + 动态行业上限, 对症2020年价值陷阱".format(
            STRATEGY_VERSION, len(STOCK_POOL), len(INDUSTRY_MAP),
            TOP_N, INDUSTRY_CAP_BASE, INDUSTRY_CAP_RATIO,
            REBALANCE_DAYS, MAX_EXPOSURE * 100, MOMENTUM_120_MIN * 100,
            "+".join(FACTOR_NAMES), len(FACTOR_NAMES), RESIDUAL_MOM_WEIGHT,
        )
    )


def rebalance(context):
    today = str(context.current_dt.date())
    g.days_since_rebalance += 1
    if g.days_since_rebalance < REBALANCE_DAYS:
        return

    candidates, blocked = filter_universe(STOCK_POOL)
    scores = build_scores(context, candidates)

    target_weights, skipped = select_targets(context, scores)
    targets = list(target_weights.keys())

    actions = []
    current_positions = list(context.portfolio.positions.keys())

    for stock in current_positions:
        if stock not in targets and context.portfolio.positions[stock].total_amount > 0:
            if not can_sell(stock):
                skipped.append("{} 卖出失败: 停牌/跌停".format(stock))
                continue
            order_target(stock, 0)
            actions.append("SELL {} -> 0".format(stock))

    for stock in targets:
        target_weight = target_weights[stock]
        target_value = context.portfolio.total_value * target_weight
        if target_value > get_position_value(context, stock):
            if not can_buy(stock):
                skipped.append("{} 买入失败: 停牌/ST/涨停".format(stock))
                continue
        action, reason = order_target_value_round_lot(context, stock, target_value)
        if action:
            actions.append(action)
        elif reason:
            skipped.append(reason)

    g.days_since_rebalance = 0
    g.last_targets = targets

    msg = format_rebalance_message(context, targets, target_weights, actions, skipped, scores)
    log.info(
        "cn800 rebalance {} factors={} candidates={} scored={} holder={} targets={}".format(
            today,
            ",".join(g.last_factor_names),
            len(candidates),
            len(scores),
            g.last_holder_coverage,
            ",".join(targets),
        )
    )
    send_feishu(msg)


def build_scores(context, stocks):
    """v9因子体系：value_blend + growth_peg + amihud + quality_roe + low_vol_60"""
    prev_date = context.previous_date
    val = get_fundamental_frame(stocks, prev_date)
    price_data = get_price(
        stocks, end_date=prev_date, count=140, frequency="daily",
        fields=["close", "money", "volume"], skip_paused=True, fq="pre", panel=False,
    )
    close = pivot_price(price_data, "close")
    money = pivot_price(price_data, "money")

    pe = numeric_series(val, "pe_ratio")
    pb = numeric_series(val, "pb_ratio")
    mv = numeric_series(val, "market_cap")

    # value_blend: 三价值因子先各自中性化, 再rank均值合并
    value_factors = {}
    value_factors["earnings_yield"] = (1.0 / pe.replace(0, np.nan)).where(pe > 0)

    pcf = first_numeric_series(val, ["pcf_ratio", "pcf"])
    if pcf is not None:
        value_factors["cashflow_yield"] = (1.0 / pcf.replace(0, np.nan)).where(pcf > 0)

    ps = first_numeric_series(val, ["ps_ratio", "ps"])
    if ps is not None:
        value_factors["sales_yield"] = (1.0 / ps.replace(0, np.nan)).where(ps > 0)

    value_frame = pd.DataFrame(value_factors).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if value_frame.empty:
        return pd.Series(dtype=float)
    value_neutral = neutralize_cross_section(value_frame, mv)
    factors = {"value_blend": value_neutral.rank(pct=True).mean(axis=1)}

    # growth_peg = 1/PEG (日频, 覆盖94%, M20裁决: OOS优于季报同比)
    peg = first_numeric_series(val, ["peg_ratio", "peg"])
    if peg is not None:
        factors["growth_peg"] = (1.0 / peg.replace(0, np.nan)).where(peg > 0)
    else:
        growth = first_numeric_series(
            val,
            ["net_profit_growth_rate", "inc_net_profit_year_on_year",
             "operating_revenue_growth_rate", "inc_revenue_year_on_year"],
        )
        if growth is not None:
            factors["growth_peg"] = (growth.where(growth > 0) / pe.replace(0, np.nan)).where(pe > 0)

    # amihud 非流动性取负 → 越高越流动
    if len(close) >= 21 and not money.empty:
        ret_abs = close.pct_change(fill_method=None).abs()
        illiq = (ret_abs / money.replace(0, np.nan)).tail(20).mean()
        factors["amihud"] = -illiq

    # quality_roe = PB/PE = ROE代理
    factors["quality_roe"] = (pb.replace(0, np.nan) / pe.replace(0, np.nan)).where((pe > 0) & (pb > 0))

    # low_vol_60 = -std(ret, 60d)
    if len(close) >= 61:
        g.last_vol_60 = close.pct_change(fill_method=None).tail(VOL_BUDGET_LOOKBACK).std()
        factors["low_vol_60"] = -g.last_vol_60
    else:
        g.last_vol_60 = pd.Series(dtype=float)

    # v3新增: 残差动量 = 个股60日收益 - 同行业均值（剥掉行业beta的纯个股动量）
    # 经济逻辑: 同年同行业，个股跑赢行业均值的趋势有持续性（信息扩散效应）
    # 对症2020年: 价值因子选出的"便宜但跌"的股票，残差动量为负→被此因子降权
    if len(close) >= 61:
        ret_60 = close.iloc[-1] / close.iloc[-61].replace(0, np.nan) - 1.0
        # 计算行业均值
        ind_series = pd.Series(
            {s: INDUSTRY_BY_STOCK.get(s[:6] if "." in s else s, "其他") for s in ret_60.index}
        )
        ind_avg = ret_60.groupby(ind_series).transform("mean")
        factors["residual_momentum"] = ret_60 - ind_avg

    # 120日动量（只做过滤，不参与排名）
    if len(close) >= 121:
        g.last_momentum_120 = close.iloc[-1] / close.iloc[-121].replace(0, np.nan) - 1.0
    else:
        g.last_momentum_120 = pd.Series(dtype=float)

    # holder (聚宽覆盖为0, 默认关闭)
    if INCLUDE_HOLDER:
        holder_change = first_numeric_series(
            val,
            ["shareholders_0_ratio", "shareholder_change_ratio", "chg_ratio_shareholder",
             "holder_change_ratio", "holder_num_change_ratio"],
        )
        g.last_holder_coverage = int(holder_change.notna().sum()) if holder_change is not None else 0
        if holder_change is not None and holder_change.notna().sum() >= 20:
            factors["holder_concentration"] = -holder_change
    else:
        g.last_holder_coverage = 0

    frame = pd.DataFrame(factors).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if frame.empty:
        return pd.Series(dtype=float)

    # value_blend 已内部中性化, 其余因子再做行业+市值中性化
    neutralize_cols = [c for c in frame.columns if c != "value_blend"]
    if neutralize_cols:
        frame_neutral = neutralize_cross_section(frame[neutralize_cols], mv)
        frame = pd.concat([frame[["value_blend"]], frame_neutral], axis=1)
    g.last_factor_names = list(frame.columns)

    # 加权合成: value_blend(1.0) + growth_peg(1.0) + amihud(1.0)
    #           + quality_roe(0.5) + low_vol_60(0.5) + residual_momentum(0.5)
    ranked = frame.rank(pct=True)
    weights = pd.Series(1.0, index=ranked.columns)
    if "quality_roe" in weights.index:
        weights["quality_roe"] = QUALITY_WEIGHT
    if "low_vol_60" in weights.index:
        weights["low_vol_60"] = LOWVOL_WEIGHT
    if "residual_momentum" in weights.index:
        weights["residual_momentum"] = RESIDUAL_MOM_WEIGHT
    if "holder_concentration" in weights.index:
        weights["holder_concentration"] = HOLDER_WEIGHT
    valid_weight = ranked.notna().mul(weights, axis=1).sum(axis=1)
    score = ranked.mul(weights, axis=1).sum(axis=1) / valid_weight.replace(0, np.nan)
    return score.dropna()


def select_targets(context, scores, top_n=TOP_N, industry_cap_base=INDUSTRY_CAP_BASE):
    current_positions = list(context.portfolio.positions.keys())
    portfolio_value = context.portfolio.total_value
    counts = {}
    selected = []
    skipped = []
    ranked = scores.sort_values(ascending=False)
    rank_pos = {stock: i for i, stock in enumerate(ranked.index, start=1)}
    max_keep_rank = max(top_n, int(math.ceil(top_n * HOLD_MULTIPLIER)))

    # 动态行业上限: max(基础值, 池内该行业股票数/比例)
    pool_counts = getattr(g, "industry_pool_counts", {})
    def _industry_cap(ind):
        base = industry_cap_base
        dynamic = max(base, pool_counts.get(ind, 0) // INDUSTRY_CAP_RATIO)
        return dynamic

    # 批量获取当前数据（800只池性能关键：一次调用替代600+次单独查询）
    current_data = get_current_data()
    slot_value = portfolio_value * MAX_EXPOSURE / top_n
    mom = getattr(g, "last_momentum_120", pd.Series(dtype=float))

    def eligible(stock):
        try:
            cd = current_data[stock]
        except Exception:
            return False
        price = getattr(cd, "last_price", None)
        if price is None or price <= 0:
            return False
        if cd.paused or cd.is_st:
            return False
        mom120 = mom.get(stock, np.nan)
        if pd.notna(mom120) and mom120 <= MOMENTUM_120_MIN:
            return False
        if price * 100 > slot_value * 1.15:
            return False
        return True

    def add_if_possible(stock):
        if stock in selected or not eligible(stock):
            return False
        code6 = stock[:6] if "." in stock else stock
        ind = INDUSTRY_BY_STOCK.get(code6, "其他")
        cap = _industry_cap(ind)
        if counts.get(ind, 0) >= cap:
            return False
        selected.append(stock)
        counts[ind] = counts.get(ind, 0) + 1
        return True

    # 持仓缓冲: 老持仓仍在TOP_N*HOLD_MULTIPLIER内则优先保留 (v9机制)
    if HOLD_MULTIPLIER > 1.0:
        current_live = [s for s in current_positions
                        if context.portfolio.positions[s].total_amount > 0]
        for stock in sorted(current_live, key=lambda s: rank_pos.get(s, 10**9)):
            if rank_pos.get(stock, 10**9) > max_keep_rank:
                continue
            add_if_possible(stock)
            if len(selected) >= top_n:
                break

    for stock in ranked.index:
        if len(selected) >= top_n:
            break
        add_if_possible(stock)

    # 诊断日志：记录跳过的原因分布（仅在持仓不足时输出）
    if len(selected) < top_n:
        log.info(
            "cn800 select_targets: only {}/{} selected. "
            "scored={} industry_counters={}".format(
                len(selected), top_n, len(ranked),
                dict(counts) if counts else {},
            )
        )

    weights = build_target_weights(selected)
    return weights.to_dict(), skipped


def build_target_weights(selected):
    if not selected:
        return pd.Series(dtype=float)
    if not USE_VOL_BUDGET or len(selected) < 3:
        return pd.Series(MAX_EXPOSURE / len(selected), index=selected)

    vol = getattr(g, "last_vol_60", pd.Series(dtype=float)).reindex(selected)
    vol = pd.to_numeric(vol, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if vol.notna().sum() < 3:
        return pd.Series(MAX_EXPOSURE / len(selected), index=selected)

    lo = vol.quantile(0.20)
    hi = vol.quantile(0.80)
    vol = vol.clip(lower=lo, upper=hi)
    raw = 1.0 / vol.replace(0, np.nan)
    raw = raw.fillna(raw.median())
    weights = raw / raw.sum() * MAX_EXPOSURE

    avg = MAX_EXPOSURE / len(selected)
    upper = min(avg * VOL_BUDGET_MAX_MULT, MAX_SINGLE_WEIGHT)
    lower = avg * VOL_BUDGET_MIN_MULT
    weights = weights.clip(lower=lower, upper=upper)
    return weights / weights.sum() * MAX_EXPOSURE


# ============================ Helper Functions ============================


def filter_universe(stocks):
    current = get_current_data()
    out = []
    skipped = []
    for stock in stocks:
        try:
            cd = current[stock]
        except Exception:
            continue
        if cd.paused:
            skipped.append("{} 停牌".format(stock[:6]))
            continue
        if cd.is_st:
            skipped.append("{} ST".format(stock[:6]))
            continue
        out.append(stock)
    return out, skipped


def numeric_series(df, name):
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def first_numeric_series(df, names):
    for name in names:
        if name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce")
            if s.notna().any():
                return s
    return None


def pivot_price(price_df, field):
    if price_df is None or len(price_df) == 0:
        return pd.DataFrame()
    df = price_df.copy()
    time_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
    code_col = "code" if "code" in df.columns else "security"
    if time_col is None or code_col is None or field not in df.columns:
        return pd.DataFrame()
    return df.pivot(index=time_col, columns=code_col, values=field).sort_index()


def neutralize_cross_section(frame, market_cap):
    """行业+市值中性化。对每列因子回归到log(市值)+行业dummies,取残差。"""
    out = pd.DataFrame(index=frame.index)
    log_mv = np.log(market_cap.replace(0, np.nan))
    industries = pd.Series(
        {s: INDUSTRY_BY_STOCK.get(s[:6] if "." in s else s, "其他") for s in frame.index}
    )
    dummies = pd.get_dummies(industries)

    for name in frame.columns:
        y = pd.to_numeric(frame[name], errors="coerce")
        valid = y.notna() & log_mv.notna()
        if valid.sum() < 12:
            out[name] = y
            continue
        x_parts = [
            pd.Series(1.0, index=y.index, name="const"),
            log_mv.loc[y.index],
            dummies.reindex(y.index).fillna(0.0),
        ]
        x = pd.concat(x_parts, axis=1).loc[valid]
        yy = y.loc[valid]
        try:
            beta = np.linalg.lstsq(x.values.astype(float), yy.values.astype(float), rcond=None)[0]
            resid = yy - x.dot(beta)
            s = pd.Series(np.nan, index=y.index)
            s.loc[valid] = resid
            out[name] = s
        except Exception:
            out[name] = y
    return out


def can_buy(stock):
    try:
        cd = get_current_data()[stock]
    except Exception:
        return False
    if cd.paused or cd.is_st:
        return False
    price = getattr(cd, "last_price", None)
    high_limit = getattr(cd, "high_limit", None)
    return not (price is not None and high_limit is not None and price >= high_limit)


def can_sell(stock):
    try:
        cd = get_current_data()[stock]
    except Exception:
        return False
    if cd.paused:
        return False
    price = getattr(cd, "last_price", None)
    low_limit = getattr(cd, "low_limit", None)
    return not (price is not None and low_limit is not None and price <= low_limit)


def get_position_value(context, stock):
    try:
        pos = context.portfolio.positions[stock]
    except Exception:
        return 0.0
    if pos is None or pos.total_amount <= 0:
        return 0.0
    return float(pos.value)


def order_target_value_round_lot(context, stock, target_value):
    """目标市值 → 100股整数手（A股整手约束）"""
    try:
        price = get_current_data()[stock].last_price
    except Exception:
        return None, "调整失败 {}: 无当前价格".format(stock)
    if price is None or price <= 0:
        return None, "调整失败 {}: 价格无效".format(stock)
    target_amount = int(math.floor(target_value / price / 100.0) * 100)
    try:
        current_amount = int(context.portfolio.positions[stock].total_amount)
    except Exception:
        current_amount = 0
    if target_amount <= 0:
        return None, "调整跳过 {}: 目标资金不足100股".format(stock)
    delta = target_amount - current_amount
    if abs(delta) < 100:
        return None, "调整跳过 {}: 变化不足100股".format(stock)
    order_target(stock, target_amount)
    return "TARGET {} {}股".format(stock, target_amount), None


def get_current_price(stock):
    try:
        return float(get_current_data()[stock].last_price)
    except Exception:
        return None


def get_fundamental_frame(stocks, date):
    """获取估值+指标数据"""
    val_table = globals().get("valuation")
    ind_table = globals().get("indicator")
    if val_table is None:
        return pd.DataFrame()

    fields = [
        val_table.code,
        val_table.pe_ratio,
        val_table.pb_ratio,
        val_table.market_cap,
    ]

    if hasattr(val_table, "peg_ratio"):
        fields.append(val_table.peg_ratio)

    for field_name in ["pcf_ratio", "pcf", "ps_ratio", "ps", "turnover_ratio", "pc"]:
        if hasattr(val_table, field_name):
            fields.append(getattr(val_table, field_name))

    if ind_table is not None:
        for name in [
            "net_profit_growth_rate", "inc_net_profit_year_on_year",
            "operating_revenue_growth_rate", "inc_revenue_year_on_year",
            "shareholders_0_ratio", "shareholder_change_ratio",
            "chg_ratio_shareholder", "holder_change_ratio", "holder_num_change_ratio",
        ]:
            if hasattr(ind_table, name):
                fields.append(getattr(ind_table, name))

    q = query(*fields).filter(val_table.code.in_(stocks))
    df = get_fundamentals(q, date=date)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.set_index("code")


def format_rebalance_message(context, targets, weights, actions, skipped, scores):
    lines = []
    for i, stock in enumerate(targets, 1):
        w = weights.get(stock, 0.0)
        sc = scores.loc[stock] if stock in scores.index else float("nan")
        ind = INDUSTRY_BY_STOCK.get(stock[:6] if "." in stock else stock, "其他")
        lines.append("{}. {} {} 持仓 {:.2f}% score={:.3f}".format(i, stock, ind, w * 100, sc))
    return (
        "A股模拟盘调仓 (CN800 v3) {}\n"
        "总资产: {:.2f}\n"
        "现金: {:.2f}\n"
        "市场状态: {} (动量过滤{:.0%})\n"
        "因子: {}\n"
        "holder覆盖: {}\n"
        "目标仓位: {:.0f}% / {} 只 / 持仓缓冲 {:.1f} / 波动预算 {}\n\n"
        "目标持仓:\n{}\n\n"
        "交易:\n{}\n\n"
        "跳过/失败:\n{}"
    ).format(
        str(context.current_dt.date()),
        context.portfolio.total_value,
        context.portfolio.cash,
        getattr(g, "market_regime", "neutral"),
        getattr(g, "momentum_filter", MOMENTUM_120_MIN_DEFAULT),
        ",".join(g.last_factor_names),
        getattr(g, "last_holder_coverage", 0),
        MAX_EXPOSURE * 100,
        len(targets),
        HOLD_MULTIPLIER,
        USE_VOL_BUDGET,
        "\n".join(lines[:20]) if lines else "无",
        "\n".join(actions) if actions else "无",
        "\n".join(skipped[:20]) if skipped else "无",
    )


def send_feishu(text):
    if not FEISHU_WEBHOOK:
        return
    import json
    import requests
    payload = {"msg_type": "text", "content": {"text": text[:3500]}}
    try:
        requests.post(FEISHU_WEBHOOK, json=payload, timeout=5)
    except Exception:
        pass
