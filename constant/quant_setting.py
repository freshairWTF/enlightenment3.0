from dataclasses import dataclass
from typing import Literal

from constant.type_ import (
    CLASS_LEVEL, GROUP_MODE, FACTOR_WEIGHT,
    FILTER_MODE, POSITION_WEIGHT, CYCLE
)


#####################################################
@dataclass
class FactorSetting:
    """因子配置信息"""
    factor_name: str                            # 因子名
    primary_classification: str                 # 一级分类
    secondary_classification: str               # 二级分类

    cycle: CYCLE = "week"                       # 周期
    half_life: int = 100                        # 半衰期，默认100期

    reverse: bool = False                       # 方向反转
    standardization: bool = True                # 标准化
    market_value_neutral: bool = True           # 市场中性化
    industry_neutral: bool = True               # 行业中性化

    restructure: bool = False                   # 因子重构
    restructure_denominator: str = ""           # 因子重构分母

    filter_mode: FILTER_MODE | None = None      # 过滤模式
    entire_filter: bool = False                 # 全部股票
    overall_filter: bool = False                # 整体股票
    large_filter: bool = False                  # 大市值
    small_filter: bool = False                  # 小市值
    mega_filter: bool = False                   # 超大市值

    bull_market: bool = False                   # 牛市
    bear_market: bool = False                   # 熊市
    shocking_market: bool = False               # 震荡市


#####################################################
@dataclass
class ModelSetting:
    """模型配置信息"""
    industry_info: dict[str, str]                                   # 行业信息
    filter_mode: FILTER_MODE                                        # 过滤模式

    position_weight_method: POSITION_WEIGHT                         # 仓位权重方法
    position_distribution: tuple[float, float]                      # 仓位集中度

    factors_setting: list[FactorSetting]                            # 因子设置
    class_level: CLASS_LEVEL = "一级行业"                             # 中性化行业
    lag_period: int = 1                                             # 滞后周期

    group_nums: int = 10                                            # 分组数
    group_mode: GROUP_MODE = "frequency"                            # 分组方法

    secondary_factor_weight_method: FACTOR_WEIGHT = "equal"         # 二级分类因子权重方法
    bottom_factor_weight_method: FACTOR_WEIGHT = "equal"            # 底层因子权重方法
    factor_weight_window: int = 12                                  # 因子权重窗口数

    factor_filter: bool = False                                     # 因子过滤
    factor_primary_classification: list[str] | None = None          # 因子一级分类
    factor_secondary_classification: list[str] | None = None        # 因子二级分类
    factor_half_life: tuple[int, int] | None = None                 # 因子半衰期区间
    factor_market: list[str] | None = None                          # 因子市场
    factor_filter_mode: list[FILTER_MODE] | None = None             # 因子过滤模式


#####################################################
@dataclass
class Trade:
    """交易数据类"""
    code: str
    time_stamp: str
    strike_price: float
    volume: float
    direction: Literal["buy", "sell"]


#####################################################
class Trader:
    """交易类"""

    @classmethod
    def buy(
            cls,
            code: str,
            time_stamp: str,
            strike_price: float,
            volume: float
    ) -> Trade:
        """买入"""
        return Trade(
            code,
            time_stamp,
            strike_price,
            volume,
            "buy"
        )

    @classmethod
    def sell(
            cls,
            code: str,
            time_stamp: str,
            strike_price: float,
            volume: float
    ) -> Trade:
        """卖出"""
        return Trade(
            code,
            time_stamp,
            strike_price,
            volume,
            "sell"
        )

    @classmethod
    def get_volume(
            cls,
            capital: float,
            price: float,
            total_volume: float
    ):
        """
        获取能够成交的数量
        :param capital: 资金
        :param price: 价格
        :param total_volume: 当期总成交量
        :return: 成交量
        """
