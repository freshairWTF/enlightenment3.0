from constant.factor_library import *
from constant.quant_setting import ModelSetting

from model.model_service import ModelAnalyzer
from model.linear_multi_factors_model import LinearMultiFactors
from model.xgboost_multi_factors_model import XGBoostMultiFactors


# --------------------------------------------
def linear_multi_factors_model():
    """线性多因子模型"""
    analyzer = ModelAnalyzer(
        model=LinearMultiFactors,
        model_setting=model_setting,
        source_dir=source_dir,
        storage_dir=storage_dir,
        cycle="month",
    )
    analyzer.run()


# --------------------------------------------
def xgboost_multi_factors_model():
    """xgboost多因子模型"""
    analyzer = ModelAnalyzer(
        model=XGBoostMultiFactors,
        model_setting=model_setting,
        source_dir=source_dir,
        storage_dir=storage_dir,
        cycle="month",
    )
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    # 路径参数
    source_dir = "202503M"
    storage_dir = "xgboost/测试1"

    # 因子参数设置
    factors_setting = list(OVERALL_FACTOR.values())

    # 模型参数设置
    model_setting = ModelSetting(
        industry_info={"全部": "三级行业"},
        filter_mode="_entire_filter",
        factors_setting=factors_setting,
        class_level="三级行业",
        lag_period=1,
        group_nums=10,
        group_mode="frequency",
        factor_weight_method="ir_decay_weight",
        factor_weight_window=12,
        dimension_reduction=False,
        orthogonal=False,
        position_weight_method="equal",
        position_distribution=(3, 1)
    )

    # 回测
    # xgboost_multi_factors_model()
    linear_multi_factors_model()
