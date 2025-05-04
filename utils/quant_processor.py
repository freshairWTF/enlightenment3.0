from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_processor import DataProcessor
from constant.type_ import GROUP_MODE, validate_literal_params
from constant.quant import RESTRUCTURE_FACTOR, PROHIBIT_MV_NEUTRAL


####################################################
class QuantProcessor:
    """量化数据处理"""

    processor = DataProcessor

    # ---------------------------------------
    # 因子预处理
    # ---------------------------------------
    @classmethod
    def preprocessing_factor_data(
            cls,
            data: dict[str, pd.DataFrame],
            factor_name: str,
            standardization: bool = True,
            market_value_neutral: bool = True,
            industry_neutral: bool = True,
            restructure: bool = False
    ) -> dict[str, pd.DataFrame]:
        """
        预处理数据
        :param data: 原始数据
        :param factor_name: 因子名
        :param standardization: 标准化
        :param market_value_neutral: 市值中性化
        :param industry_neutral: 行业中性化
        :param restructure: 因子重构
        :return: 预处理因子数据
        """
        def __process_single_date(
                df_: pd.DataFrame
        ) -> pd.DataFrame:
            """单日数据处理"""
            processed_col = f"processed_{factor_name}"
            df_[processed_col] = df_[factor_name]

            # -1 重构因子
            if restructure and RESTRUCTURE_FACTOR.get(factor_name, ""):
                df_[processed_col] = cls.processor.restructure_factor(
                    df_[processed_col],
                    df_[RESTRUCTURE_FACTOR.get(factor_name)]
                )

            # -2 第一次 去极值、标准化
            df_[processed_col] = cls.processor.percentile(df_[processed_col])
            if standardization:
                df_[processed_col] = cls.processor.standardization(df_[processed_col])

            # -3 中性化
            if market_value_neutral and factor_name not in PROHIBIT_MV_NEUTRAL:
                df_[processed_col] = cls.processor.market_value_neutral(df_[processed_col], df_["对数市值"])

            if industry_neutral:
                df_[processed_col] = cls.processor.industry_neutral(df_[processed_col], df_["行业"])

            # -4 第二次 去极值、标准化
            df_[processed_col] = cls.processor.percentile(df_[processed_col])
            if standardization:
                df_[processed_col] = cls.processor.standardization(df_[processed_col])

            return df_

        processed_data = {}
        for date, df in data.items():
            try:
                processed_data[date] = __process_single_date(df)
            except ValueError:
                continue
        return processed_data

    @classmethod
    def preprocessing_factors_by_setting(
            cls,
            data: dict[str, pd.DataFrame],
            factors_setting: list[dataclass]
    ) -> dict[str, pd.DataFrame]:
        """
        预处理方法
        :param data: 原始数据
        :param factors_setting: 因子配置
        :return: 预处理好的数据
        """
        def __process_single_date(
                df_: pd.DataFrame,
        ) -> pd.DataFrame:
            """单日数据处理"""
            for setting in factors_setting:
                factor_name = setting.factor_name
                processed_col = f"processed_{factor_name}"
                df_[processed_col] = df_[factor_name]

                # 1- 重构因子
                if setting.restructure:
                    df_[processed_col] = cls.processor.restructure_factor(
                        df_[processed_col],
                        df_[setting.restructure_denominator]
                    )

                # -2 顺序反转
                if setting.reverse:
                    df_[processed_col] = df_[processed_col] * -1

                # -3 第一次 去极值、标准化
                df_[processed_col] = cls.processor.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = cls.processor.standardization(df_[processed_col])

                # -4 中性化
                if setting.market_value_neutral:
                    df_[processed_col] = cls.processor.market_value_neutral(df_[processed_col], df_["对数市值"])
                if setting.industry_neutral:
                    df_[processed_col] = cls.processor.industry_neutral(df_[processed_col], df_["行业"])

                # -5 第二次 去极值、标准化
                df_[processed_col] = cls.processor.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = cls.processor.standardization(df_[processed_col])

            return df_

        processed_data = {}
        for date, df in data.items():
            try:
                processed_data[date] = __process_single_date(df)
            except ValueError:
                continue

        return processed_data

    # ---------------------------------------
    # 分组
    # ---------------------------------------
    @classmethod
    @validate_literal_params
    def divide_into_group(
            cls,
            data: dict[str, pd.DataFrame],
            factor_col: str,
            processed_factor_col: str,
            group_mode: GROUP_MODE,
            group_nums: int,
            group_label: list[str],
            negative: bool = False
    ) -> dict[str, pd.DataFrame]:
        """
        分组 -1等距 distant；-2 等频 frequency
        :param data: 分组数据
        :param factor_col: 分组因子列名
        :param processed_factor_col: 分组预处理因子列名
        :param group_mode: 分组模式
        :param group_nums: 分组数
        :param group_label: 分组标签
        :param negative: 负值单列
        """
        method = {
            "distant": cls.__distant,
            "frequency": cls.__frequency
        }.get(group_mode, cls.__frequency)

        result: dict[str, pd.DataFrame] = {}
        for date, df in data.items():
            try:
                if df.shape[0] >= group_nums:
                    df["group"] = method(
                        df, factor_col, processed_factor_col, group_nums, group_label, negative
                    )
                    result[date] = df
            except ValueError:
                continue
        return result

    @staticmethod
    def __distant(
            df: pd.DataFrame,
            factor_col: str,
            processed_factor_col: str,
            group_nums: int,
            group_label: list[str],
            negative: bool
    ) -> pd.Series:
        """等距分组"""
        if negative:
            negative_mask = df[factor_col] < 0
            df["group"] = np.where(
                negative_mask,
                'negative',
                pd.NA
            )
            non_negative_group = pd.cut(
                df.loc[~negative_mask, processed_factor_col],
                bins=group_nums,
                labels=group_label,
                duplicates="drop"
            )
            df.loc[~negative_mask, "group"] = non_negative_group
            return df["group"].astype("category")
        else:
            return pd.cut(
                pd.to_numeric(df[processed_factor_col]),
                bins=group_nums,
                labels=group_label,
                duplicates="drop"
            )

    @staticmethod
    def __frequency(
            df: pd.DataFrame,
            factor_col: str,
            processed_factor_col: str,
            group_nums: int,
            group_label: list[str],
            negative: bool
    ) -> pd.Series:
        """等频分组"""
        if negative:
            negative_mask = df[factor_col] < 0
            df["group"] = np.where(
                negative_mask,
                'negative',
                pd.NA
            )
            non_negative_group = pd.qcut(
                df.loc[~negative_mask, processed_factor_col],
                q=group_nums,
                labels=group_label,
                duplicates="drop"
            )
            df.loc[~negative_mask, "group"] = non_negative_group
            return df["group"].astype("category")
        else:
            return pd.qcut(
                pd.to_numeric(df[processed_factor_col]),
                q=group_nums,
                labels=group_label,
                duplicates="drop"
            )

    # ---------------------------------------
    # 平移
    # ---------------------------------------
    @classmethod
    def shift_factors(
            cls,
            raw_data: dict[str, pd.DataFrame],
            lag_periods: int,
            factors_col: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        # 按日期排序（假设输入键均为交易日）
        sorted_dates = sorted(raw_data.keys(), key=lambda x: pd.to_datetime(x))

        # 构建日期对：(原始日期, 滞后日期)
        date_pairs = list(zip(sorted_dates, sorted_dates[lag_periods:]))

        # 初始化结果字典
        result = {}

        # 处理每个日期对
        for date, lag_date in date_pairs:
            # 获取原始数据（浅拷贝）
            src_df = raw_data[date].copy(deep=False)
            # 筛选有效因子列
            if factors_col is None:
                factors_col = src_df.columns.difference(["pctChg"]).tolist()
            valid_factors = [f for f in factors_col if f in src_df.columns]
            # 创建或更新目标日期的数据
            if lag_date not in result:
                # 获取目标日期的原始数据（含 pctChg）
                target_original_df = raw_data[lag_date].copy(deep=False)
                # 更新因子列
                target_original_df[valid_factors] = src_df[valid_factors]
                result[lag_date] = target_original_df.dropna(how="any")

        return result

    # @classmethod
    # def shift_factors(
    #         cls,
    #         raw_data: dict[str, pd.DataFrame],
    #         lag_periods: int,
    #         factors_col: list[str] | None = None,
    # ) -> dict[str, pd.DataFrame]:
    #     """
    #     将每个DataFrame中的因子向后移指定期数，用于 T-N期因子 与 T期涨跌幅 的拟合回归
    #     :param raw_data: 原始数据
    #     :param lag_periods: 滞后期数
    #     :param factors_col: 因子名
    #     :return: 平移后的数据
    #     """
    #     # 深拷贝原始数据
    #     copied_data = {k: v.copy(deep=True) for k, v in raw_data.items()}
    #
    #     # 确定需要平移的列
    #     if factors_col is None:
    #         sample_df = next(iter(copied_data.values()))
    #         factors_col = sample_df.columns.difference(["pctChg"]).tolist()
    #     else:
    #         factors_col = [f for f in factors_col if f != "pctChg"]
    #
    #     # 无有效列直接返回
    #     if not factors_col:
    #         return copied_data
    #
    #     # 合并所有数据并添加临时日期标记
    #     combined = pd.concat(
    #         {date: df[factors_col] for date, df in copied_data.items()},
    #         names=["date"]
    #     ).reset_index(level="date")
    #
    #     # 按资产分组，滞后平移
    #     shifted = combined.groupby(combined.index)[factors_col].shift(lag_periods)
    #     combined[factors_col] = shifted
    #
    #     # 按日期拆分回字典
    #     result = {}
    #     sorted_dates = sorted(copied_data.keys(), key=lambda x: pd.to_datetime(x))
    #     for date in sorted_dates:
    #         df = combined[combined.date == date].drop(columns="date")
    #         df = df.reindex(copied_data[date].index)
    #
    #         result_date = copied_data[date].copy()
    #         result_date[factors_col] = df[factors_col]
    #         result_date = result_date.dropna(subset=factors_col, how="any")
    #
    #         if not result_date.empty:
    #             result[date] = result_date
    #
    #     return result

    # ---------------------------------------
    # 正交化
    # ---------------------------------------
    @classmethod
    def calc_symmetric_orthogonal(
            cls,
            processed_data: dict[str, pd.DataFrame],
            selected_factors: dict[str, list[str]]
    ) -> dict[str, pd.DataFrame]:
        """
        对称正交化
        :param processed_data: 预处理过的数据
        :param selected_factors: 每期选取的因子
        :return: 对称正交后的数据
        """
        return {
            date: (
                df.copy()
                .assign(**cls.processor.symmetric_orthogonal(df[selected_factors[date]]))
            )
            for date, df in processed_data.items()
        }
