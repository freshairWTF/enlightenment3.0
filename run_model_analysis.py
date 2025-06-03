from constant.factor_library import *
from constant.quant_setting import ModelSetting

from model.model_service import ModelAnalyzer
from model.linear_multi_factors_model import LinearMultiFactors
from model.xgboost_multi_factors_model import XGBoostMultiFactors
from model.randomforest_multi_factors_model import RandomForestMultiFactors


# -----------------------------
# 模型
# -----------------------------
MODEL = {
    "linear": LinearMultiFactors,
    "xgboost": XGBoostMultiFactors,
    "randomforest": RandomForestMultiFactors
}


# --------------------------------------------
def model_backtest():
    """模型回测"""
    analyzer = ModelAnalyzer(
        model=MODEL[model_setting.model],
        model_setting=model_setting,
        source_dir=source_dir,
        storage_dir=storage_dir,
        cycle=model_setting.cycle,
    )
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    # 路径参数
    source_dir = "20250530-WEEK-跟踪"
    storage_dir = "模型跟踪/linear-20250530W-ir衰减-旧因子库"

    """
    增加波动率因子
    寻找高波动的
    
    增加成长因子
    
    pca降维前后因子数不一致怎么办？
    
    阿尔法因子   直接ir加权
    弱/风险因子  使用ir衰减   
    """
    # 模型参数设置
    model_setting = ModelSetting(
        # 模型/周期/因子
        model="linear",
        cycle="week",
        factors_setting=list(FACTOR_LIBRARY.values()),

        # 目标股票池
        industry_info={"全部": "三级行业"},
        filter_mode="_entire_filter",

        # 目标因子
        factor_filter=False,
        # factor_primary_classification=["基本面因子"],
        # factor_secondary_classification=["质量因子"],
        factor_filter_mode=["_entire_filter"],
        # factor_half_life=(3, 6),

        # 因子处理方法
        class_level="三级行业",
        bottom_factor_weight_method="ir_decay_weight",
        secondary_factor_weight_method="ir_decay_weight",
        factor_weight_window=12,

        # 分组
        group_nums=20,
        group_mode="frequency",

        # 仓位
        position_weight_method="group_long_only",
        position_distribution=(3, 1),
    )

    # 回测
    model_backtest()
