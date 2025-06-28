from typing import get_origin, get_args, Literal, get_type_hints
from functools import wraps
from inspect import signature


# 财务周期
FINANCIAL_CYCLE = Literal["year", "half", "quarter"]
# 财务报表标签
FINANCIAL_SHEET = Literal['BS', 'PS', 'CF']

# k线周期
KLINE_CYCLE = Literal["year", "half", "quarter", "month", "week", "day", "original_day"]
# k线复权标签
KLINE_SHEET = Literal["backward_adjusted", "split_adjusted", "unadjusted"]

# 一般周期
CYCLE = Literal["year", "half", "quarter", "month", "week", "day"]
# 分析维度
DIMENSION = Literal["mac", "meso", "micro"]
# 宏观/中观合成权重
WEIGHTS = Literal["等权", "市值", "营业收入", "净利润"]

# 行业分类表sheet
INDUSTRY_SHEET = Literal["A", "Total_A", "AHU"]
# 行业分类表返回数据类型
INDUSTRY_RETURN_TYPE = Literal["dataframe", "dict", "list"]
# 行业分类表分类列名
CLASS_LEVEL = Literal["一级行业", "二级行业", "三级行业", "美林时钟", "库存周期", "自定义", "自定义一"]

# 分组模式
GROUP_MODE = Literal["frequency", "distant"]
# 过滤模式
FILTER_MODE = Literal[
                '_white_filter', '_entire_filter', '_overall_filter',
                '_mega_cap_filter', '_large_cap_filter', '_small_cap_filter'
            ]

# 因子权重
FACTOR_WEIGHT = Literal[
    "equal",
    "ic_weight", "ir_weight",
    "ir_decay_weight", "ir_decay_weight_with_diff_halflife"
]
# 仓位权重
POSITION_WEIGHT = Literal["equal", "group_equal","long_only", "group_long_only", "hedge", "group_hedge"]
# 收益率计算模式
CALC_RETURN_MODE = Literal["equal", "mv_weight", "position_weight"]

# 错误处理方式
ERROR: Literal["raise", "warn", "ignore"] = "raise"


#######################################################
def validate_literal_params(func):
    """装饰器：检查函数中形参为 Literal 的参数是否合法"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        type_hints = get_type_hints(func)

        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            if param_name in type_hints:
                type_hint = type_hints[param_name]
                origin = get_origin(type_hint)

                # 只处理Literal类型
                if origin is Literal:
                    literal_args = get_args(type_hint)
                    if param_value not in literal_args:
                        valid_options = ", ".join(map(repr, literal_args))
                        raise ValueError(
                            f"Invalid value for parameter '{param_name}': {param_value!r}. "
                            f"Valid options are: {valid_options}"
                        )
        return func(*args, **kwargs)

    return wrapper
