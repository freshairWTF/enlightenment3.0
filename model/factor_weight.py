import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from type_ import validate_literal_params, FACTOR_WEIGHT
from utils.processor import DataProcessor


####################################################
class FactorWeight:
    """因子权重"""

    processor = DataProcessor()

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

    @classmethod
    def _get_ir_decay_weight_by_halflife(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            half_life: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ir自适应衰减加权权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param half_life: 因子半衰期
        :return 因子权重
        """
        # -1 计算 rank ic
        rank_ic = cls._calc_rank_ics(factors_data, factors_name)

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
    @classmethod
    @validate_literal_params
    def get_factors_weights(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            method: FACTOR_WEIGHT,
            window: int,
            half_life: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        获取因子权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param method: 权重方法
        :param window: 因子滚动均值窗口数
        :param half_life: 因子半衰期
        :return 因子权重
        """
        handlers = {
            "equal": lambda: cls._get_equal_weight(factors_name),
            "ic_weight": lambda: cls._get_ic_weight(factors_data, factors_name, window),
            "ir_weight": lambda: cls._get_ir_weight(factors_data, factors_name, window),
            "ir_decay_weight": lambda: cls._get_ir_decay_weight(factors_data, factors_name, window),
            "ir_decay_weight_by_halflife": lambda: cls._get_ir_decay_weight_by_halflife(
                factors_data,
                factors_name,
                half_life
            )
        }

        return handlers[method]()

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
