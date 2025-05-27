from constant.factor_library import *
from constant.quant_setting import ModelSetting

from model.model_service import ModelAnalyzer
from model.linear_multi_factors_model import LinearMultiFactors
from model.xgboost_multi_factors_model import XGBoostMultiFactors


# -----------------------------
# 模型
# -----------------------------
MODEL = {
    "linear": LinearMultiFactors,
    "xgboost": XGBoostMultiFactors
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
    source_dir = "20250502-WEEK-混合"
    storage_dir = "模型debug/xgboost-20250503W-ir衰退加权-滚动12期-alpha因子库"

    """
    linear-传统-ir_decay_weight
    linear-阿尔法-ir_decay_weight
    linear-阿尔法-半衰期
    
    xgboost-传统-ir_decay_weight
    xgboost-阿尔法-ir_decay_weight
    xgboost-阿尔法-半衰期
    xgboost模型有问题
    """
    # 模型参数设置
    model_setting = ModelSetting(
        # 模型/周期/因子
        model="xgboost",
        cycle="week",
        factors_setting=list(ALPHA_FACTOR_LIBRARY.values()),

        # 目标股票池
        industry_info={"全部": "三级行业"},
        filter_mode="_entire_filter",

        # 目标因子
        factor_filter=True,
        # factor_primary_classification=["基本面因子"],
        # factor_secondary_classification=["行为金融因子"],
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
