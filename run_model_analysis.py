

from quant.model_service import ModelAnalyzer
from quant_setting import ModelSetting
from model.multi_factors_model import MultiFactors

from constant.factor_library import *


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
    storage_dir = "ir衰减测试/ir加权-滚动6-夏普修改"

    # 因子参数设置
    factors_setting = list(OVERALL_FACTOR.values())
    # factors_setting = list(BATTERY_CHEMICAL_FACTOR.values())

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
        position_weight_method="equal"
    )

    # 回测
    multi_factors_model()
