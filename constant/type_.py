from typing import get_origin, get_args, Literal, get_type_hints
from functools import wraps
from inspect import signature


FINANCIAL_CYCLE = Literal["year", "half", "quarter"]
FINANCIAL_SHEET = Literal['BS', 'PS', 'CF']

KLINE_CYCLE = Literal["year", "half", "quarter", "month", "week", "day", "original_day"]
KLINE_SHEET = Literal["backward_adjusted", "split_adjusted", "unadjusted"]

CYCLE = Literal["year", "half", "quarter", "month", "week", "day"]
DIMENSION = Literal["mac", "meso", "micro"]
WEIGHTS = Literal["等权", "市值", "营业收入", "净利润"]

INDUSTRY_SHEET = Literal["A", "Total_A", "AHU"]
INDUSTRY_RETURN_TYPE = Literal["dataframe", "dict", "list"]

CLASS_LEVEL = Literal["一级行业", "二级行业", "三级行业", "美林时钟", "库存周期", "自定义", "自定义一"]
GROUP_MODE = Literal["frequency", "distant"]
FILTER_MODE = Literal[
                '_white_filter', '_entire_filter', '_overall_filter',
                '_mega_cap_filter', '_large_cap_filter', '_small_cap_filter'
            ]

FACTOR_WEIGHT = Literal["equal", "ic_weight", "ir_weight", "ir_decay_weight"]

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
