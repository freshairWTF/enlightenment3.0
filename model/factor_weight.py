import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from type_ import validate_literal_params, FACTOR_WEIGHT
from data_processor import DataProcessor


####################################################
class FactorWeight:
    """因子权重"""

    # --------------------------
    # 辅助方法
    # --------------------------
    @staticmethod
    def _calc_ir(
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
    def __calc_rank_ic(
            df: pd.DataFrame,
            factors: list[str]
    ) -> pd.Series:
        """计算当期 Rank IC"""
        rank_ic = {}

        for factor in factors:
            valid_df = df[[factor, "pctChg"]].dropna()
            if len(valid_df) < 2:
                corr = np.nan
            else:
                corr = spearmanr(
                    valid_df[factor], valid_df["pctChg"]
                ).correlation
            rank_ic[factor] = corr

        return pd.Series(rank_ic)

    @classmethod
    def _calc_rank_ics(
            cls,
            factors_data,
            factors_name
    ) -> pd.DataFrame:
        """计算Rank IC"""
        return pd.concat([
            cls.__calc_rank_ic(df, factors_name[date]).rename(date)
            for date, df in factors_data.items()
        ], axis=1).T

    # --------------------------
    # 权重方法
    # --------------------------
    @classmethod
    def _get_equal_weight(
            cls,
            factors_name: dict[str, list[str]],
    ) -> pd.DataFrame:
        """
        等权权重
        :param factors_name: T期因子名
        """
        weights = pd.concat([
            pd.Series(
                1 / len(factors),
                index=factors,
                name=date
            ) for date, factors in factors_name.items()
        ], axis=1).T

        return weights

    @classmethod
    def _get_ic_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            window: int
    ) -> pd.DataFrame:
        """
        ic加权权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param window: 滚动窗口数
        :return 因子权重
        """
        # -1 计算 rank ic
        rank_ic = cls._calc_rank_ics(factors_data, factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 计算滚动值
        rolling_ic = rank_ic_shifted.rolling(window, min_periods=1).mean().abs()

        # -4 计算权重
        weights = rolling_ic.div(rolling_ic.sum(axis=1), axis="index")

        return weights

    @classmethod
    def _get_ir_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            window: int
    ) -> pd.DataFrame:
        """
        ir加权权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param window: 滚动窗口数
        :return 因子权重
        """
        # -1 计算 rank ic
        rank_ic = cls._calc_rank_ics(factors_data, factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 计算滚动 ir
        rolling_ir = rank_ic_shifted.rolling(window, min_periods=1).apply(cls._calc_ir)

        # -4 计算权重
        weights = rolling_ir.div(rolling_ir.sum(axis=1), axis="index")

        return weights

    @classmethod
    def _get_ir_decay_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            window: int
    ) -> pd.DataFrame:
        """
        ir自适应衰减加权权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param window: 半衰期
        :return 因子权重
        """
        # -1 计算 rank ic
        rank_ic = cls._calc_rank_ics(factors_data, factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 计算自适应衰减ir
        ic_ewm_mean = rank_ic_shifted.ewm(halflife=window, min_periods=1).mean()
        ic_ewm_std = rank_ic_shifted.ewm(halflife=window, min_periods=1).std()
        rolling_ir = (ic_ewm_mean / ic_ewm_std).abs().replace(np.inf, 0).fillna(0)

        # -4 计算权重
        weights = rolling_ir.div(rolling_ir.sum(axis=1), axis="index")

        return weights

    # --------------------------
    # 公开 API
    # --------------------------
    @classmethod
    @validate_literal_params
    def get_factors_weights(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            method: FACTOR_WEIGHT,
            window: int,
    ) -> pd.DataFrame:
        """
        获取因子权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param method: 权重方法
        :param window: 因子滚动均值窗口数
        :return 因子权重
        """
        handlers = {
            "equal": lambda: cls._get_equal_weight(factors_name),
            "ic_weight": lambda: cls._get_ic_weight(factors_data, factors_name, window),
            "ir_weight": lambda: cls._get_ir_weight(factors_data, factors_name, window),
            "ir_decay_weight": lambda: cls._get_ir_decay_weight(factors_data, factors_name, window)
        }

        return handlers[method]()

    @classmethod
    def synthesis_factor(
            cls,
            bottom_factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            weights: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        合成因子
        :param bottom_factors_data: 底层因子数据
        :param factors_name: T期因子名
        :param weights: 权重
        :return: 综合因子
        """
        # -1 计算综合因子
        z_score = {
            date: filtered_df.rename("综合Z值").to_frame()
            for date, df in bottom_factors_data.items()
            if not (
                filtered_df := (
                    df[factors_name[date]].mean(axis=1)
                ) if weights.empty
                else (
                        df[factors_name[date]] * weights.loc[date]
                ).sum(axis=1, skipna=True)
            ).dropna().empty
        }

        # -2 综合因子标准化
        z_score = {
            date: processed_df
            for date, df in z_score.items()
            if not (
                processed_df := DataProcessor.standardization(df, error="ignore").dropna()
            ).empty
        }

        return z_score
