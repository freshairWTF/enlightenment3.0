import numpy as np
import pandas as pd

from base_metrics import Metrics, depends_on
from constant.type_ import CYCLE, validate_literal_params
from kline_determination import KlineDetermination
from utils.data_processor import DataProcessor

"""
标记半衰期
根据半衰期筛选因子  得到 长、中、短期的选股集合

公司治理指标
行业指标

"""
###############################################################
class GovernanceMetrics(Metrics, KlineDetermination):
    """
    公司治理指标
    指标命名规则：指标名_计算窗口
    """

    @validate_literal_params
    def __init__(
            self,
            shareholders: pd.DataFrame,
            circulating_shareholders: pd.DataFrame,
            methods: dict[str, list],
            function_map: dict[str, str]
    ):
        """
        :param shareholders: 股东数据
        :param circulating_shareholders: 流通股东数据
        :param methods: 需要实现的方法
        :param function_map: 已定义的方法对应方法名
        """
        self.shareholders_data = self._merger_data(shareholders, circulating_shareholders)
        print(self.shareholders_data.columns)
        print(function_map)
        print(methods)
        # print(dd)
        self.metrics = pd.DataFrame()

        self.function_map = function_map
        self.methods = methods

    # --------------------------
    # 初始化数据处理 方法
    # --------------------------
    @staticmethod
    def _merger_data(
            shareholders: pd.DataFrame,
            circulating_shareholders: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        数据合并：十大股东 + 十大流通股东
        :param shareholders: 股东数据
        :param circulating_shareholders: 流通股东数据
        :return 合并后的数据
        """
        return pd.concat(
            [
                shareholders,
                circulating_shareholders.add_prefix("流通_")
            ],
            axis=1
        )

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def calculate(
            self
    ) -> None:
        """计算接口"""
        for method, windows in self.methods.items():
            if method in self.function_map:
                method_name = self.function_map.get(method)
                if method_name and hasattr(self, method_name):
                    getattr(self, method_name)
                else:
                    raise ValueError(f"未实现的方法: {method}")
            else:
                raise ValueError(f"未定义的指标: {method}")

    # --------------------------
    # 量价 私有方法
    # --------------------------
    def _shareholders_cr10(
            self
    ) -> None:
        """前十大股东合计占比"""
        print(self.shareholders_data)

        print(dd)

        # df =
        #
        # self.metrics[f"异常换手率_{window}"] = df
