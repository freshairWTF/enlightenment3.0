from typing import Literal
from functools import wraps

import numpy as np
import pandas as pd

from constant.type_ import CYCLE, validate_literal_params
from constant.quant import ANNUALIZED_DAYS


###############################################################
def depends_on(*dependencies):
    """指标依赖装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for dep in dependencies:
                if dep not in self.metrics.columns:
                    getattr(self, self.function_map[dep])()
            return func(self, *args, **kwargs)  # 传递所有参数到原函数
        return wrapper
    return decorator


###############################################################
class Metrics:

    # --------------------------
    # 计算 私有方法
    # --------------------------
    @classmethod
    def _safe_divide(
            cls,
            numerator: pd.Series,
            denominator: pd.Series,
            default: float = 0.0,
            percentage: bool = False,
            join: Literal["outer", "inner", "left", "right"] = "inner"
    ) -> pd.Series:
        """
        安全除法计算（对齐/除零/无效运算）
        :param numerator: 分子
        :param denominator: 分母
        :param default: 默认值
        :param percentage: 是否转换为百分比
        :param join: 索引对齐方式，可选 "inner", "outer", "left", "right"（默认 "inner"）
        """
        # 对齐索引 默认对齐索引
        numerator_aligned, denominator_aligned = numerator.align(denominator, join=join, axis=0)

        # 忽略 除零/无效运算
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denominator_aligned != 0, numerator_aligned / denominator_aligned, default)
        if percentage:
            result *= 100
        return pd.Series(result, index=numerator_aligned.index)
    
    @classmethod
    def _calc_rolling(
            cls,
            data: pd.DataFrame | pd.Series,
            window: int,
            min_periods: int,
            method: Literal["mean", "median", "sum"] = "mean"
    ) -> pd.Series:
        """
        通用滚动计算
        :param data: 数据
        :param window: 窗口数
        :param min_periods: 最小周期
        :param method: 方法
        :return: 滚动数据 
        """
        def _process_series(series: pd.Series) -> pd.Series:
            if method == "mean":
                return series.rolling(window=window, min_periods=min_periods).mean()
            elif method == "sum":
                return series.rolling(window=window, min_periods=min_periods).sum()
            elif method == "median":
                return series.rolling(window=window, min_periods=min_periods).median()
            else:
                raise ValueError(f"不支持该方法: {method}")

        if isinstance(data, pd.DataFrame):
            return data.apply(_process_series)
        elif isinstance(data, pd.Series):
            return _process_series(data)
        else:
            raise TypeError("仅支持 pandas DataFrame/Series 类型输入")

    @classmethod
    def _calc_growth_rate(
            cls,
            data: pd.DataFrame | pd.Series,
            window: int = 1,
            compound: bool = False
    ) -> pd.Series:
        """
        通用增长率计算（同比/环比）
        :param data: 原始数据
        :param window: 窗口数
        :param compound: 是否为复合增速
        :return: 增长率
        """
        def _process_series(series: pd.Series) -> pd.Series | float:
            filtered_series = series.dropna()
            if filtered_series.empty:
                if compound:
                    return np.nan
                else:
                    return pd.Series(np.nan, filtered_series.index)

            if compound:
                # 确保窗口起始值>0且结束值非负
                if filtered_series[0] <= 0 and filtered_series[-1] < 0:
                    return np.nan
                with np.errstate(divide="ignore", invalid="ignore"):
                    return (filtered_series[-1] / filtered_series[0]) ** (1 / len(filtered_series)) - 1

            return cls._safe_divide(
                filtered_series.diff(window), filtered_series.shift(window).abs(), percentage=True
            )

        if isinstance(data, pd.DataFrame):
            return data.apply(_process_series)
        elif isinstance(data, pd.Series):
            return _process_series(data)
        else:
            raise TypeError("仅支持 pandas DataFrame/Series 类型输入")

    @classmethod
    def _calc_proportion(
            cls,
            data: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame | pd.Series:
        """
        占比计算
        :param data: 原始数据
        :return: 占比
        """
        def _process_series(series: pd.Series) -> pd.Series:
            return series / series.sum() * 100

        if isinstance(data, pd.DataFrame):
            return data.apply(_process_series)
        elif isinstance(data, pd.Series):
            return _process_series(data)
        else:
            raise TypeError("仅支持 pandas DataFrame/Series 类型输入")

    # --------------------------
    # 其他 私有方法
    # --------------------------
    @classmethod
    @validate_literal_params
    def _setup_window(
            cls,
            cycle: CYCLE,
    ) -> int:
        """获取窗口数"""
        return ANNUALIZED_DAYS.get(cycle, np.nan)
