from dataclasses import dataclass
from typing import Literal

from constant.type_ import CLASS_LEVEL, GROUP_MODE, FACTOR_WEIGHT, FILTER_MODE


#####################################################
@dataclass
class FactorSetting:
    """因子配置信息"""
    factor_name: str                            # 因子名

    reverse: bool = False                       # 方向反转
    standardization: bool = True                # 标准化
    market_value_neutral: bool = True           # 市场中性化
    industry_neutral: bool = True               # 行业中性化

    restructure: bool = False                   # 因子重构
    restructure_denominator: str = ""           # 因子重构分母

    entire_filter: bool = False                 # 全部股票
    overall_filter: bool = False                # 整体股票
    large_filter: bool = False                  # 大市值
    small_filter: bool = False                  # 小市值
    mega_filter: bool = False                   # 超大市值


#####################################################
@dataclass
class ModelSetting:
    """模型配置信息"""
    industry_info: dict[str, str]                       # 行业信息
    filter_mode: FILTER_MODE                            # 过滤模式

    position_weight_method: str                         # 仓位权重方法

    factors_setting: list[FactorSetting]                # 因子设置
    class_level: CLASS_LEVEL = "一级行业"                 # 中性化行业
    orthogonal: bool = False                            # 正交化
    lag_period: int = 1                                 # 滞后周期

    group_nums: int = 10                                # 分组数
    group_mode: GROUP_MODE = "frequency"                # 分组方法

    factor_weight_method: FACTOR_WEIGHT = "equal"       # 因子权重方法
    factor_weight_window: int = 12                      # 因子权重窗口数


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
