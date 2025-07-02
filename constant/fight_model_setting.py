"""模型设置"""

from model import *
from fight_factor_library import *
from constant.quant_setting import ModelSetting


# -----------------------------
# 模型工厂
# -----------------------------
MODEL = {
    "traditionalLinearReg": LinearRegressionTraditionalModel,
}


# ----------------------------------------------------------
LINEAR_MODEL_SETTING = ModelSetting(
    # 模型/周期/因子
    model="traditionalLinearReg",
    cycle="week",
    factors_setting=list(FIGHT_LINEAR_FACTOR.values()),
    industry_info={"全部": "三级行业"},
    generate_trade_file=True,

    # 因子处理方法
    class_level="一级行业",
    bottom_factor_weight_method="ir_decay_weight",
    secondary_factor_weight_method="ir_decay_weight",
    factor_weight_window=12,

    # 分组
    group_nums=20,
    group_mode="frequency",
)
