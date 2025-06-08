"""计算因子权重"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from type_ import validate_literal_params, FACTOR_WEIGHT
from utils.processor import DataProcessor


####################################################
class FactorWeight:
    """因子权重"""

    processor = DataProcessor()

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

    @classmethod
    def synthesis_factor(
            cls,
            bottom_factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            weights: pd.DataFrame,
            synthesis_factor_name: str
    ) -> pd.DataFrame:
        """
        合成因子
        :param bottom_factors_data: 底层因子数据
        :param factors_name: T期因子名
        :param weights: 权重
        :param synthesis_factor_name: 合成因子名
        :return: 综合因子
        """
        df_list = []
        for date, df in bottom_factors_data.items():
            # 计算因子值
            filtered_series = (
                (df[factors_name[date]] * weights.loc[date]).sum(axis=1, skipna=True)
            ).dropna()

            if not filtered_series.empty:
                # 转换为DataFrame并添加日期列
                temp_df = filtered_series.rename(synthesis_factor_name).to_frame()
                temp_df.insert(0, 'date', date)                 # 在首列插入日期
                df_list.append(temp_df.reset_index())           # 把索引转为列

        # 纵向拼接所有数据
        factors_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

        # 标准化处理
        factors_df[synthesis_factor_name] = factors_df.groupby('date')[synthesis_factor_name].transform(
            lambda x: cls.processor.dimensionless.standardization(x, error="ignore")
        )

        return factors_df
