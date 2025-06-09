"""计算因子权重"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from type_ import validate_literal_params, FACTOR_WEIGHT


####################################################
class FactorWeight:
    """因子权重"""

    def __init__(
            self,
            factors_value: pd.DataFrame,
            factors_name: list[str],
            method: FACTOR_WEIGHT,
    ):
        """
        :param factors_value: 因子数据
        :param factors_name: 因子名列表
        :param method: 计算因子权重方法
        """
        self.factors_value = factors_value
        self.factors_name = factors_name
        self.method = method

    # --------------------------
    # ic/ir 方法
    # --------------------------
    @staticmethod
    def calc_icir(
            ic: pd.Series
    ) -> float:
        """
        计算ir
        :param ic: ic值
        :return ir值
        """
        ic_clean = ic.dropna()
        if len(ic_clean) < 2:
            return 0.0
        return np.abs(ic_clean.mean() / ic_clean.std()) if ic_clean.std() != 0 else 0.0

    @staticmethod
    def calc_rank_ic(
            factors_value: pd.DataFrame,
            factors_name: list[str]
    ) -> pd.DataFrame:
        """计算Rank IC"""
        result_dfs = []
        for date, group in factors_value.groupby("date"):
            result_dfs.append(
                pd.Series(
                    {
                        factor: spearmanr(
                            group[factor], group["pctChg"]
                        ).correlation
                        for factor in factors_name
                    },
                    name=date
                )
            )
        return pd.concat(result_dfs, axis=1).T

    # --------------------------
    # 权重方法
    # --------------------------
    def _calc_equal_weight(
            self
    ) -> pd.DataFrame:
        """
        等权权重
        :return 等权权重
        """
        result_dfs = []
        for date, group in self.factors_value.groupby("date"):
            result_dfs.append(
                pd.Series(
                    1 / len(self.factors_name),
                    index=self.factors_name,
                    name=date
                )
            )

        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    def _calc_ic_weight(
            self,
            window: int
    ) -> pd.DataFrame:
        """
        计算ic权重
        :param window: 取均值滚动窗口数
        :return ic权重
        """
        # -1 rank ic
        rank_ic = self.calc_rank_ic(self.factors_value, self.factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 滚动均值的绝对值 ic
        rolling_ic = rank_ic_shifted.rolling(window, min_periods=1).mean().abs()

        # -4 权重
        weights = rolling_ic.div(rolling_ic.sum(axis=1), axis="index")

        return weights

    def _calc_ic_decay_weight(
            self,
            window: int
    ) -> pd.DataFrame:
        """
        计算ic指数加权权重
        :param window: 取均值滚动窗口数
        :return ic权重
        """
        # -1 rank ic
        rank_ic = self.calc_rank_ic(self.factors_value, self.factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 滚动指数加权均值的绝对值 ic
        rolling_ic = rank_ic_shifted.ewm(halflife=window, min_periods=1).mean().abs()

        # -4 权重
        weights = rolling_ic.div(rolling_ic.sum(axis=1), axis="index")

        return weights

    def _calc_ir_weight(
            self,
            window: int
    ) -> pd.DataFrame:
        """
        计算ir权重
        :param window: 滚动窗口数
        :return 因子权重
        """
        # -1 rank ic
        rank_ic = self.calc_rank_ic(self.factors_value, self.factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 滚动均值的绝对值 ir
        rolling_ir = rank_ic_shifted.rolling(window, min_periods=1).apply(self.calc_icir)

        # -4 权重
        weights = rolling_ir.div(rolling_ir.sum(axis=1), axis="index")

        return weights

    def _calc_ir_decay_weight(
            self,
            window: int
    ) -> pd.DataFrame:
        """
        计算ir指数加权权重
        :param window: 半衰期
        :return 因子权重
        """
        # -1 rank ic
        rank_ic = self.calc_rank_ic(self.factors_value, self.factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 滚动均值的绝对值 ir
        ic_ewm_mean = rank_ic_shifted.ewm(halflife=window, min_periods=1).mean()
        ic_ewm_std = rank_ic_shifted.ewm(halflife=window, min_periods=1).std()
        rolling_ir = (ic_ewm_mean / ic_ewm_std).abs().replace(np.inf, 0).fillna(0)

        # -4 权重
        weights = rolling_ir.div(rolling_ir.sum(axis=1), axis="index")

        return weights

    def _calc_ir_decay_weight_with_diff_halflife(
            self,
            half_life: pd.DataFrame
    ) -> pd.DataFrame:
        """
        不同半衰期的ir指数加权权重
        :param half_life: 因子半衰期
        :return 因子权重
        """
        # -1 rank ic
        rank_ic = self.calc_rank_ic(self.factors_value, self.factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # 逐列计算指数加权指标
        def ewm_calculator(col, func):
            factor = col.name
            return col.ewm(
                halflife=half_life.loc["half_life", factor],
                min_periods=1
            ).agg(func)

        # -3 计算自适应衰减ir
        ic_ewm_mean = rank_ic_shifted.apply(lambda x: ewm_calculator(x, 'mean'))
        ic_ewm_std = rank_ic_shifted.apply(lambda x: ewm_calculator(x, 'std'))
        rolling_ir = (ic_ewm_mean / ic_ewm_std).abs().replace(np.inf, 0).fillna(0)

        # -4 计算权重
        weights = rolling_ir.div(rolling_ir.sum(axis=1), axis="index")

        return weights

    # --------------------------
    # 公开 API
    # --------------------------
    @validate_literal_params
    def get_factors_weights(
            self,
            window: int,
            half_life: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        获取因子权重
        :param window: 因子滚动均值窗口数
        :param half_life: 因子半衰期
        :return 因子权重
        """
        handlers = {
            "equal": lambda: self._calc_equal_weight(),
            "ic_weight": lambda: self._calc_ic_weight(window),
            "ic_decay_weight": lambda: self._calc_ic_decay_weight(window),
            "ir_weight": lambda: self._calc_ir_weight(window),
            "ir_decay_weight": lambda: self._calc_ir_decay_weight(window),
            "_calc_ir_decay_weight_with_diff_halflife":
                lambda: self._calc_ir_decay_weight_with_diff_halflife(half_life)
        }

        return handlers[self.method]()
