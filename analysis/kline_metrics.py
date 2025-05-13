from scipy.stats import linregress

import numpy as np
import pandas as pd

from base_metrics import Metrics, depends_on
from constant.type_ import CYCLE, validate_literal_params
from kline_determination import KlineDetermination
from utils.data_processor import DataProcessor


###############################################################
class KLineMetrics(Metrics, KlineDetermination):
    """
    量价指标计算器
    指标命名规则：指标名_计算窗口
    """

    @validate_literal_params
    def __init__(
            self,
            kline_data: pd.DataFrame,
            cycle: CYCLE,
            methods: dict[str, list],
            function_map: dict[str, str],
            index_data: pd.DataFrame | None = None,
    ):
        """
        :param kline_data: K线数据
        :param index_data: 指数K线数据
        :param cycle: 周期
        :param methods: 需要实现的方法
        :param function_map: 已定义的方法对应方法名
        """
        self.metrics = kline_data
        self.index_data = index_data
        self.cycle = cycle
        self.function_map = function_map
        self.methods = methods

        self.annual_window = self._setup_window(self.cycle)

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
                    for window in windows:
                        getattr(self, method_name)(window)
                else:
                    raise ValueError(f"未实现的方法: {method}")
            else:
                raise ValueError(f"未定义的指标: {method}")

    # --------------------------
    # 量价 私有方法（无窗口） 列名=因子名
    # --------------------------
    def _tr(
            self,
            window: int | None = None
    ) -> None:
        """真实波幅 = max(最大值,昨日收盘价) − min(最小值,昨日收盘价)"""
        highest = np.maximum(self.metrics["high"], self.metrics["preclose"])
        lowest = np.minimum(self.metrics["low"], self.metrics["preclose"])
        self.metrics["真实波幅"] = highest - lowest

    # --------------------------
    # 量价 私有方法
    # --------------------------
    def _market_beta(
            self,
            window: int
    ) -> None:
        """滚动计算市场贝塔系数"""
        if self.index_data is None:
            return
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        # 对齐数据
        stock_ret = self.metrics["pctChg"]
        index_ret = self.index_data["pctChg"]
        aligned_stock, aligned_index = stock_ret.align(index_ret, join="inner")

        # 计算市场贝塔
        covariance = aligned_stock.rolling(rolling_window).cov(aligned_index)
        market_var = aligned_index.rolling(rolling_window).var()
        beta = self._safe_divide(
            covariance,
            market_var
        )

        self.metrics[f"市场贝塔_{window}"] = beta.rename(f"market_beta_{window}")

    def _abnormal_turn(
            self,
            window: int
    ) -> None:
        """异常换手率指标 = 第t期的换手率 / t1...t12期的换手率累加和"""
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        turn_series = self.metrics["turn"]
        abnormal = self._safe_divide(
            turn_series,
            turn_series.rolling(rolling_window).sum(),
            percentage=True
        )

        self.metrics[f"异常换手率_{window}"] = abnormal

    def _accumulated_yield(self, window):
        """
        累加收益率：n期累加收益率
        """
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        self.metrics[f"累加收益率_{window}"] = (
            self._calc_rolling(
                self.metrics["pctChg"],
                rolling_window,
                rolling_window,
                "sum"
            )
        )

    # def _ma_close(
    #         self,
    #         window: int
    # ) -> None:
    #     """收盘价均线"""
    #     # 滚动窗口
    #     rolling_window = int(window * self.annual_window)
    #
    #     self.metrics[f"收盘价均线_{window}"] = (
    #         self._calc_rolling(
    #             self.metrics["close"],
    #             rolling_window,
    #             rolling_window
    #         )
    #     )

    def _ma_close(
            self,
            window: int
    ) -> None:
        """收盘价均线"""
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        self.metrics[f"收盘价均线_{window}"] = (
            self._calc_rolling(
                self.metrics["close"],
                rolling_window,
                rolling_window
            )
        )

    def _ma_volume(
            self,
            window: int
    ) -> None:
        """成交量均线"""
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        self.metrics[f"成交量均线_{window}"] = (
            self._calc_rolling(
                self.metrics["volume"],
                rolling_window,
                rolling_window
            )
        )

    def _ma_turn(
            self,
            window: int
    ) -> None:
        """换手率均线"""
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        self.metrics[f"换手率均线_{window}"] = (
            self._calc_rolling(
                self.metrics["turn"],
                rolling_window,
                rolling_window
            )
        )

    @depends_on("真实波幅")
    def _ma_atr(
            self,
            window: int
    ) -> None:
        """真实波幅均线"""
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        self.metrics[f"真实波幅均线_{window}"] = (
            self._calc_rolling(
                self.metrics["真实波幅"],
                rolling_window,
                rolling_window
            )
        )

    def _std_yield(
            self,
            window: int
    ) -> None:
        """收益率标准差"""
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        self.metrics[f"收益率标准差_{window}"] = (
            self.metrics["pctChg"]
            .rolling(window=rolling_window)
            .std()
        )

    def _std_turn(
            self,
            window: int
    ) -> None:
        """换手率标准差"""
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        self.metrics[f"换手率标准差_{window}"] = (
                self.metrics["turn"]
                .rolling(window=rolling_window)
                .std()
        )

    @depends_on("真实波幅")
    def _std_atr(
            self,
            window: int
    ) -> None:
        """
        真实波幅 = max(最大值,昨日收盘价) − min(最小值,昨日收盘价)
        """
        # 滚动窗口
        rolling_window = int(window * self.annual_window)

        self.metrics[f"std_atr_{window}"] = (
            self.metrics["真实波幅"]
            .rolling(window=rolling_window)
            .std()
        )

    def _slope(
            self,
            window: int
    ) -> None:
        """计算价格序列的线性趋势斜率"""
        def calculate_slope(window_values):
            valid_mask = ~np.isnan(window_values)
            valid_count = valid_mask.sum()
            if valid_count < 2:
                return np.nan
            x = np.arange(len(window_values))[valid_mask]       # 生成时间序列自变量（使用位置索引）
            y = window_values[valid_mask]
            result = linregress(x, y)
            return result.slope

        # 滚动窗口
        rolling_window = int(window * self.annual_window)
        # 标准化
        price_series = DataProcessor.standardization(self.metrics["close"])

        # 滚动计算斜率
        slopes = price_series.rolling(
            window=rolling_window,
            min_periods=2
        ).apply(calculate_slope, raw=False)

        self.metrics[f"斜率_{window}"] = slopes

    def _limit_up_number(
            self
    ):
        """涨停次数"""

    def _limit_down_number(
            self
    ):
        """跌停次数"""

    """
    涨停次数
    波动率
    资金流入
    """

    def _ch_relay_form(
            self,
            window: int
    ) -> None:
        """
        来源 -> 川海
        上涨中继形态：
            1、2根中大阳线爆量，5%-8%、成交量放大3-5倍；
            2、第三根上影线缩量，影线长度约1/3-1/2；

        超参：
            1、中大阳线定义：5%
            2、成交量均值窗口数：30
            3、成交量最低涨幅：3倍
            4、上影线相对实体区间：0.5-1.2
            5、成交量最低缩量：0.8
        """
        # -1 2根中大阳线爆量
        # medium_or_large_positive_line = self.n_consecutive_mask(
        #     self.positive_line(
        #         self.metrics,
        #         min_change=5
        #     ),
        #     2
        # )

        # -1 1根中大阳线爆量
        medium_or_large_positive_line = self.positive_line(
            self.metrics,
            min_change=5
        )
        explosive_quantity = self.explosive_quantity(
            self.metrics,
            window=30,
            min_change=3
        )
        condition_1 = (medium_or_large_positive_line & explosive_quantity).shift(1).fillna(False)

        # -2 第三根上影线缩量
        upper_shadow_p = self.positive_line(
            self.metrics,
            upper_shadow_bounds=(0.5, 1.2)
        )
        upper_shadow_n = self.negative_line(
            self.metrics,
            upper_shadow_bounds=(0.5, 1.2)
        )
        reduced_quantity = self.reduced_quantity(
            self.metrics,
            window=2,
            min_change=0.9
        )
        condition_2 = (upper_shadow_p | upper_shadow_n) & reduced_quantity

        self.metrics["ch_relay_form"] = (condition_1 & condition_2).astype("int")
