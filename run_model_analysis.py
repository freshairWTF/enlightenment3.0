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
市场环境分析
不同市场下的收益率表现
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
    "ZScoreLinearReg": LinearRegressionZScoreModel,
    "ZScorePCLinearReg": LinearRegressionPCModel,
    "ZScoreHigherZeroLinearReg": LinearRegressionHigherZeroModel,
    "level2linearReg": LinearRegressionLevel2Model,
    "testLineaReg": LinearRegressionTestModel,

    "xgboostReg": XGBoostRegressionModel,
    "xgboostCVReg": XGBoostRegressionCVModel,

    "xgboostCla": XGBoostClassificationModel,
    "xgboostClaNS": XGBoostClassificationNSModel,
    "xgboostClaCV": XGBoostClassificationCVModel,
    "xgboostClaNSCV": XGBoostClassificationNSCVModel,

    "randomForestReg": RandomForestClassificationModel,
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
    source_dir = "模型因子测试集"
    storage_dir = "测试/ZScoreLinearReg"

    filter_mode: FILTER_MODE = "_white_filter"
    # 模型参数设置
    model_setting = ModelSetting(
        # 模型/周期/因子
        model="ZScoreLinearReg",
        cycle="week",
        factors_setting=list(FACTOR_LIBRARY.values()),
        industry_info={"全部": "三级行业"},

        # 目标因子
        factor_filter=False,

        # 因子处理方法
        class_level="一级行业",
        bottom_factor_weight_method="ir_decay_weight",
        secondary_factor_weight_method="ir_decay_weight",
        factor_weight_window=12,

        # 分组
        group_nums=20,
        group_mode="frequency",
    )

    # 回测
    model_backtest()
