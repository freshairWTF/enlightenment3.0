from constant.factor_library import *
from constant.quant_setting import ModelSetting

from model.model_service import ModelAnalyzer
from model.linear_multi_factors_model import LinearMultiFactors
from model.xgboost_multi_factors_model import XGBoostMultiFactors


# --------------------------------------------
def linear_multi_factors_model(cycle_):
    """线性多因子模型"""
    analyzer = ModelAnalyzer(
        model=LinearMultiFactors,
        model_setting=model_setting,
        source_dir=source_dir,
        storage_dir=storage_dir,
        cycle=cycle_,
    )
    analyzer.run()


# --------------------------------------------
def xgboost_multi_factors_model(cycle_):
    """xgboost多因子模型"""
    analyzer = ModelAnalyzer(
        model=XGBoostMultiFactors,
        model_setting=model_setting,
        source_dir=source_dir,
        storage_dir=storage_dir,
        cycle=cycle_,
    )
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    # 路径参数
    source_dir = "20250503-WEEK"
    storage_dir = "模型跟踪/linear-20250503W-全部股票-20组-ir衰退加权-滚动12期-等权仓位"

    # 因子参数设置
    cycle = "week"
    factors_setting = list(FACTOR_LIBRARY.values())

    # 模型参数设置
    model_setting = ModelSetting(
        industry_info={"全部": "三级行业"},
        filter_mode="_entire_filter",

        factors_setting=factors_setting,
        class_level="三级行业",
        lag_period=1,

        group_nums=20,
        group_mode="frequency",

        factor_weight_method="ir_decay_weight",
        factor_weight_window=12,

        position_weight_method="group_equal",
        position_distribution=(1, 1)
    )

    # 回测
    # xgboost_multi_factors_model(cycle)
    linear_multi_factors_model(cycle)
