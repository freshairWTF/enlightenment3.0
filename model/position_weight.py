import numpy as np
import pandas as pd

from data_processor import DataProcessor
from type_ import validate_literal_params, POSITION_WEIGHT


####################################################
class PositionWeight:
    """仓位权重"""

    # --------------------------
    # 辅助方法
    # --------------------------
    @staticmethod
    def __get_standardized_feature_ranking_vector(
            factors_data: dict[str, pd.DataFrame],
            factor_name: str,
            feature_range: tuple[int, int]
    ) -> dict[str, pd.Series]:
        """
        获取标准化特征排名向量
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param feature_range: 映射空间：-1 (0, 1)纯多头；-2 (-1, 1)多空对冲
        :return 标准化特征排名向量
        """
        return {
            date: DataProcessor.normalization(
                df[factor_name].rank(),
                feature_range=feature_range
            )
            for date, df in factors_data.items()
        }

    @staticmethod
    def __get_group_standardized_feature_ranking_vector(
            factors_data: dict[str, pd.DataFrame],
            factor_name: str,
            group_column: str,
            feature_range: tuple[int, int]
    ) -> dict[str, pd.Series]:
        """
        获取标准化特征排名向量（分组）
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param group_column: 分组列名
        :param feature_range: 映射空间：-1 (0, 1)纯多头；-2 (-1, 1)多空对冲
        :return 标准化特征排名向量
        """
        result = {}
        for date, df in factors_data.items():
            # 分组
            grouped = df.groupby(group_column, group_keys=False)
            # 组内排序
            ranked = grouped[factor_name].rank()
            # 组内标准化到目标范围
            result[date] = ranked.groupby(df[group_column]).transform(
                lambda x: DataProcessor.normalization(x, feature_range)
            )

        return result

    @staticmethod
    def __get_power_sorting_weights(
            ranking_vector: dict[str, pd.Series | pd.DataFrame],
            p: float,
            q: float,
            group: bool = False
    ) -> dict[str, pd.Series]:
        """
        幂排序权重（按日期处理每个横截面）
        :param ranking_vector: 行索引为日期，列名为股票代码，值为标准化排名分数
        :param p: 正得分幂参数（控制多头权重集中度）
                p > 1	        强化头部集中度	            押注少数高得分标的
                p = 1	        线性权重（原样保留得分比例）	平衡策略
                0 < p < 1	    分散权重	                降低头部集中度，避免过度暴露
                p = 0	        等权重（需特殊处理）
        :param q: 负得分幂参数（控制空头权重集中度）
        :return: 幂排序权重
        """
        # 定义单日期权重计算函数
        def _calculate_weights(row: pd.Series) -> pd.Series:
            # 提取当前日期的所有股票分数
            scores = row.values
            stock_codes = row.index

            # 分离正、负得分
            positive_mask = scores > 0
            negative_mask = scores < 0

            # 计算分母（正负得分的幂和）
            sum_pos = np.sum(scores[positive_mask] ** p) if np.any(positive_mask) else 0.0
            sum_neg = np.sum(np.abs(scores[negative_mask]) ** q) if np.any(negative_mask) else 0.0

            # 计算每个股票的权重
            weights = []
            for s in scores:
                if s < 0:
                    weight = - (np.abs(s) ** q) / sum_neg if sum_neg != 0 else 0.0
                elif s == 0:
                    weight = 0.0
                else:
                    weight = (s ** p) / sum_pos if sum_pos != 0 else 0.0
                weights.append(weight)

            return pd.Series(weights, index=stock_codes)

        if group:
            # 按日期、分组计算权重
            return {
                date: df.groupby(
                    "group",
                    group_keys=False
                )["score"].apply(
                    _calculate_weights
                ).rename("position_weight")
                for date, df in ranking_vector.items()
            }
        else:
            # 按日期逐行计算权重
            return {
                date: _calculate_weights(
                    vector
                ).rename("position_weight")
                for date, vector in ranking_vector.items()
            }

    @classmethod
    def _get_equal_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.Series]:
        """
        等权权重
        :param factors_data: 因子数据
        """
        weight = {
            date: pd.Series(
                1 / df.shape[0],
                index=df.index
            ).rename("position_weight")
            for date, df in factors_data.items()
        }

        return weight

    @classmethod
    def _get_hedge_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factor_name: str,
            distribution: tuple[float, float]
    ) -> dict[str, pd.Series]:
        """
        对冲仓位权重
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param distribution: 权重分布集中度
        :return: 对冲仓位权重
        """
        # -1 标准化特征排名向量
        ranking_vector = cls.__get_standardized_feature_ranking_vector(
            factors_data,
            factor_name,
            feature_range=(-1, 1)
        )

        # -2 计算幂排序权重
        weight = cls.__get_power_sorting_weights(
            ranking_vector,
            distribution[0],
            distribution[1]
        )

        return weight

    @classmethod
    def _get_group_hedge_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factor_name: str,
            distribution: tuple[float, float]
    ) -> dict[str, pd.Series]:
        """
        纯多头仓位权重
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param distribution 权重分布集中度
        :return: 纯多头仓位权重
        """
        # -1 标准化特征排名向量
        ranking_vector = cls.__get_group_standardized_feature_ranking_vector(
            factors_data,
            factor_name,
            group_column="group",
            feature_range=(-1, 1)
        )

        # -2 合并 分组信息
        ranking_vector = {
            date: pd.concat([
                vector.rename("score"),
                factors_data[date]["group"]
            ], axis=1)
            for date, vector in ranking_vector.items()
        }

        # -3 计算幂排序权重
        weight = cls.__get_power_sorting_weights(
            ranking_vector,
            distribution[0],
            distribution[1],
            group=True
        )

        return weight

    @classmethod
    def _get_long_only_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factor_name: str,
            distribution: tuple[float, float]
    ) -> dict[str, pd.Series]:
        """
        纯多头仓位权重
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param distribution 权重分布集中度
        :return: 纯多头仓位权重
        """
        # -1 标准化特征排名向量
        ranking_vector = cls.__get_standardized_feature_ranking_vector(
            factors_data,
            factor_name,
            feature_range=(0, 1)
        )

        # -2 计算幂排序权重
        weight = cls.__get_power_sorting_weights(
            ranking_vector,
            distribution[0],
            distribution[1]
        )

        return weight

    @classmethod
    def _get_group_long_only_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factor_name: str,
            distribution: tuple[float, float]
    ) -> dict[str, pd.Series]:
        """
        纯多头仓位权重
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param distribution 权重分布集中度
        :return: 纯多头仓位权重
        """
        # -1 标准化特征排名向量
        ranking_vector = cls.__get_group_standardized_feature_ranking_vector(
            factors_data,
            factor_name,
            group_column="group",
            feature_range=(0, 1)
        )

        # -2 合并 分组信息
        ranking_vector = {
            date: pd.concat([
                vector.rename("score"),
                factors_data[date]["group"]
            ], axis=1)
            for date, vector in ranking_vector.items()
        }

        # -3 计算幂排序权重
        weight = cls.__get_power_sorting_weights(
            ranking_vector,
            distribution[0],
            distribution[1],
            group=True
        )

        return weight

    # --------------------------
    # 公开 API
    # --------------------------
    @classmethod
    @validate_literal_params
    def get_weights(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factor_name: str,
            method: POSITION_WEIGHT,
            distribution: tuple[float, float] = (1, 1)
    ) -> dict[str, pd.Series]:
        """
        获取仓位权重
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param method: 权重方法
        :param distribution 权重分布集中度
                p > 1	        强化头部集中度	            押注少数高得分标的
                p = 1	        线性权重（原样保留得分比例）	平衡策略
                0 < p < 1	    分散权重	                降低头部集中度，避免过度暴露
                p = 0	        等权重（需特殊处理）
        :return 仓位权重
        """
        handlers = {
            "equal": lambda: cls._get_equal_weight(factors_data),
            "long_only": lambda: cls._get_long_only_weight(factors_data, factor_name, distribution),
            "group_long_only": lambda: cls._get_group_long_only_weight(factors_data, factor_name, distribution),
            "hedge": lambda: cls._get_hedge_weight(factors_data, factor_name, distribution),
            "group_hedge": lambda: cls._get_group_hedge_weight(factors_data, factor_name, distribution)
        }

        return handlers[method]()
