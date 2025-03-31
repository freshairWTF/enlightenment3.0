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
    storage_dir = "ir衰减测试/测试-纯头多-3"

    # 因子参数设置
    factors_setting = list(OVERALL_FACTOR.values())
    # factors_setting = list(BATTERY_CHEMICAL_FACTOR.values())
    """
    多因子怎么处理
    量价因子
    特殊因子
    """
    # 模型参数设置
    model_setting = ModelSetting(
        # industry_info={"正极": "自定义", "负极": "自定义", "电解液": "自定义", "隔膜": "自定义"},
        industry_info={"全部": "三级行业"},
        filter_mode="_overall_filter",
        factors_setting=factors_setting,
        class_level="二级行业",
        orthogonal=True,
        lag_period=1,
        group_nums=10,
        group_mode="frequency",
        factor_weight_method="ir_weight",
        factor_weight_window=6,
        position_weight_method="group_long_only",
        position_distribution=(3, 1)
    )

    # 回测
    multi_factors_model()
