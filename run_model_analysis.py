"""
模型回测
"""
"""
    -2 barra中性化 9大类风格因子
    -4 基学习器的选择 以及调参
    -1 模型集成 集成 堆叠
    -2 风格因子中性化
    -5 因子跟踪（除 估值价差 之外）
    -6 数据 行业分类表 以及 各类数据的完善
    -7 标的池
    
    -9 样本内外区分                                                      70%
    -3 扩张窗口因子 -> 历史分位数回测
    -17 席位因子群 -> 期货数据纳入体系
            回测设计 -> 股指席位因子 强弱分组 做截面
                      席位因子 做时序
            爬虫 -> future_seat/ic/date                               100%
            清洗                                                      20%
    -15 交易结果分析/风格剖析                                         70%                                                
    
    -16 因子池边际贡献模块
    -12 策略容量 每只个股可以成交当日成交额的5%，若当日无法完成交易，则再下一日继续尝试
    -18 level2数据的信息挖掘
    -19 回归模型 + 分类模型 回归模型阈值控制 -> 仓位管理
    
    龙珠 时序的回归器 + 截面的回归器
    
    时序逻辑与截面逻辑要统一
    或者能不能策略器自带总权重 
    或者说 使用股指/指数与全市场合成数据做回归？分类？
    
    起始回归器大于0的比例就是自带的权重计算器/
    目标为大于0的分类器也是自带权重计算器 
    截面回归器 线性 -> 等频分组 top组 
    截面回归器 线性 -> 大于0等频分组 top组  
    截面回归器 非线性 -> 等频分组 top组 
    截面回归器 非线性 -> 大于0等频分组 top组
    截面分类器 分类 -> top组 
    截面分类器 分类 -> 等频分组 top组
    
    择时器
    6个策略器 可以通过集成 -> 3个策略器
    
    量化多头策略 
        -1 多策略器 集成/堆叠
        -2 单一策略器的设计
   
"""


"""
商品日内
300多个 降维后40-50个
持股数量 80-100 选股3000+
65% 财务因子
25% 量价
剩余 另类数据因子
80%线性
20%非线性
年换手40-50倍

微盘 + 择时 + 优选
实盘！！！
因子等权 75量价 25基本面 70反转 25-30动量
市场择时 -> 仓位管理 标的池变换
平均市值 65亿元
单体1% 一SW级行业20%
年化双边换手60-80倍

指增 跟踪误差 超额
相关性分组
海外营收、国央企
流动性3000+
成分股至少40-50%
GAN 对抗网络生成模型
升维再降维 ？？二次项？？？pca/合成？？？ ——> 树模型
每个半小时预测一次

2千个因子 -> 神经网络 -> zscore
模型 -1 子模型 -2 参数
模型样本内/外IC IR  t检验 因为它都是合成一个zscore
80%+量价 基本面因子作为条件，可以理解为因子池的划分，然后再细化量价的处理
模型组合 等权/  抽样等权？
风控
"""

from constant.factor_library import *
from constant.quant_setting import ModelSetting

from model import *
from model.model_service import ModelAnalyzer
from constant.type_ import FILTER_MODE


# -----------------------------
# 模型工厂
# -----------------------------
MODEL = {
    "linearReg": LinearRegressionModel,
    "traditionalLinearReg": LinearRegressionTraditionalModel,
    "linearTestReg": LinearRegressionTestModel,
    "xgboostReg": XGBoostRegressionModel,
    "xgboostCVReg": XGBoostRegressionCVModel,
    "xgboostCla": XGBoostClassificationModel,
    "randomForestReg": RandomForestClassificationModel,

    "fightLinearReg": FightLinearRegressionModel,
    "fightLinearHigherReg": FightLinearRegressionHigherModel,
}


# --------------------------------------------
def model_backtest():
    """模型回测"""
    analyzer = ModelAnalyzer(
        model=MODEL[model_setting.model],
        model_setting=model_setting,
        filter_mode=filter_mode,
        source_dir=source_dir,
        storage_dir=storage_dir,
        cycle=model_setting.cycle,
    )
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    # 路径参数
    source_dir = "barra因子"
    storage_dir = "小市值专题/超级小盘股-市值+低波动+股价"

    # filter_mode: FILTER_MODE = "_entire_filter"
    filter_mode: FILTER_MODE = "_small_cap_filter"
    # 模型参数设置
    model_setting = ModelSetting(
        # 模型/周期/因子
        model="traditionalLinearReg",
        cycle="week",
        factors_setting=list(FACTOR_TEST.values()),
        industry_info={"全部": "三级行业"},

        # 目标因子
        factor_filter=False,
        factor_filter_mode=["_entire_filter"],
        # factor_primary_classification=["基本面因子"],
        # factor_secondary_classification=["质量因子"],
        # factor_half_life=(3, 6),

        # 因子处理方法
        class_level="一级行业",
        bottom_factor_weight_method="equal",
        secondary_factor_weight_method="ir_decay_weight",
        factor_weight_window=12,

        # 分组
        group_nums=10,
        group_mode="frequency",

        # 仓位
        position_weight_method="group_long_only",
        position_distribution=(3, 1),
    )

    # 回测
    model_backtest()
