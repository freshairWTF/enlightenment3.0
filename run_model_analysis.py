from constant.factor_library import *
from constant.quant_setting import ModelSetting
from model.model_service import ModelAnalyzer
from model.multi_factors_model import MultiFactors


# --------------------------------------------
def multi_factors_model():
    """多因子模型"""
    analyzer = ModelAnalyzer(
        model=MultiFactors,
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
    storage_dir = "新增量价/全部因子-ir_decay加权-等权"

    # 因子参数设置
    factors_setting = list(FACTOR_LIBRARY.values())
    """
    多因子怎么处理
    特殊因子
    """
    # 模型参数设置
    model_setting = ModelSetting(
        industry_info={"全部": "三级行业"},
        filter_mode="_entire_filter",
        factors_setting=factors_setting,
        class_level="三级行业",
        orthogonal=True,
        lag_period=1,
        group_nums=10,
        group_mode="frequency",
        factor_weight_method="ir_decay_weight",
        factor_weight_window=12,
        position_weight_method="equal",
        position_distribution=(3, 1)
    )

    # 回测
    multi_factors_model()
