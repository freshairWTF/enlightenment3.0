"""模型工具"""
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

import pandas as pd
import statsmodels.api as sm

from type_ import validate_literal_params, POSITION_WEIGHT, FACTOR_WEIGHT
from utils.processor import DataProcessor


processor = DataProcessor()


###################################################
class ModelUtils:

    def __init__(self):
        self.extract = ModelExtract
        self.dimension = DimensionalityReduction
        self.synthesis = FactorSynthesis
        self.pos_weight = PositionWeight
        self.factor_weight = FactorWeight


###################################################
class ModelExtract:

    @staticmethod
    def get_factors_synthesis_table(
            factors_setting: list,
            top_level: bool = True
    ) -> dict[str, list[str]]:
        """
        获取一级分类因子构成字典
        :param factors_setting: 因子设置
        :param top_level: 是否为一级分类因子
        :return
        """
        result = defaultdict(list)
        for setting in factors_setting:
            if top_level:
                result[setting.primary_classification].append(setting.secondary_classification)
            else:
                result[setting.secondary_classification].append(f"processed_{setting.factor_name}")
        return {k: list(dict.fromkeys(v)) for k, v in result.items()}

    @staticmethod
    def get_factor_setting_metric(
            factors_setting: list,
            metric_name: str
    ) -> pd.DataFrame:
        """
        获取因子设置指标
        :param factors_setting: 因子设置
        :param metric_name: 指标名
        :return 因子设置指标
        """
        result = {}
        for setting in factors_setting:
            result[setting.factor_name]= [setting.getattr(metric_name)]
        return pd.DataFrame(result, index=[metric_name]).add_prefix("processed_")


###################################################
class DimensionalityReduction:
    """因子降维"""

    @staticmethod
    def vif_reduction(
            input_df: pd.DataFrame,
            factors_name: list[str],
            vif_threshold: float = 10,
    ) -> dict[hash, list[str]]:
        """
        基于VIF，逐步回归筛选因子
            终止条件1：仅剩截距项时退出循环（len(variables) <= 1）。
            终止条件2：剩余变量的VIF最大值低于阈值（vif.max() < self.vif_threshold）。
        :param input_df: 因子数据
        :param factors_name: 因子名
        :param vif_threshold: VIF筛选阈值，高于此值的因子被剔除
        :return 无多重共线性的因子
        """
        def _vif_process(df: pd.DataFrame) -> list[str]:

            # 添加截距项
            df = sm.add_constant(df)
            # 变量名
            variables = df.columns.tolist()

            while True:
                # 终止条件1：仅剩const
                if len(variables) <= 1:
                    break

                vif = pd.Series(
                    [
                        variance_inflation_factor(df[variables].values, i)
                        for i in range(len(variables))
                    ],
                    index=variables
                )

                # 终止条件2：剩余变量VIF均低于阈值
                if vif.max() < vif_threshold:
                    break

                # 找到并剔除VIF最高的变量（排除截距项）
                candidates = vif.drop('const', errors='ignore')
                if candidates.empty:
                    break

                drop_var = str(candidates.idxmax())
                variables.remove(drop_var)

            # 剔除截距项
            variables.remove("const")

            return variables

        return {
            date: _vif_process(group[factors_name])
            for date, group in input_df.groupby("date")
        }

    @staticmethod
    def pca(
            input_df: pd.DataFrame,
            factors_synthesis_table: dict[str, list[str]],
            n_components: float = 0.95
    ) -> pd.DataFrame:
        """
        PCA降维
        :param input_df: 因子数据
        :param factors_synthesis_table: 因子合成字典
        :param n_components: 解释方差
        :return 降维因子
        """
        result = []
        for date, df in input_df.groupby("date"):
            # ------------------------------
            # 截面PCA降维
            # ------------------------------
            section_df = pd.DataFrame()
            for senior, component_factors in factors_synthesis_table.items():
                # -1 二级因子分类PCA降维
                pca = PCA(n_components=n_components)
                pca_result = pd.DataFrame(
                    pca.fit_transform(df[component_factors]),
                    index=df.index,
                    columns=[f"{senior}{i+1}" for i in range(pca.n_components_)]
                )
                pca_result[["index", "date"]] = df[["index", "date"]]

                # -2 截面数据合并
                if section_df.empty:
                    section_df = pca_result
                else:
                    section_df = section_df.merge(pca_result, on=['index', 'date'], how='inner')

            # -3 添加数据集
            result.append(section_df)

        # ------------------------------
        # T期截面数据合并
        # ------------------------------
        return pd.concat(result, ignore_index=True)

    # ------------------------------------------
    # 公开 API方法
    # ------------------------------------------
    # def fit_transform(
    #         self,
    #         processed_data: dict[str, pd.DataFrame],
    # ) -> dict[str, pd.DataFrame]:
    #     """
    #     处理全部日期数据
    #     :param processed_data: 输入数据 {date: df}
    #     :return: 降维后的数据
    #     """
    #     """
    #     1、因子回测 需要检测出 因子的非线性特征
    #               标记 尽管ic不过关，但是icir大于0.5的弱因子，用以合成增强因子（在因子回测时加入因子合成模块）
    #     2、因子合成 u型因子加入二项式
    #     3、模型 非线性模型 主要是xgboost与随机森林
    #     """
    #     # -1 三级底层因子正交
    #     processed_data = self._bottom_factors_orthogonal(processed_data)
    #
    #     # -2 二级因子合成
    #     secondary_factors_df = self._synthesis_secondary_factor(
    #         processed_data
    #     )
    #
    #     # -3 二级因子降维
    #     pca_df = self._pca(secondary_factors_df)
    #
    #     # -4 转换为 {date: pd.DataFrame}格式
    #     result = {
    #         str(date): group.set_index('index').drop("date", axis=1).dropna(axis=1, how="all").dropna(how="any")
    #         for date, group in pca_df.groupby('date')
    #     }
    #
    #     return result


###################################################
class FactorSynthesis:
    """因子合成"""

    @staticmethod
    def factors_orthogonal(
            input_df: pd.DataFrame,
            factors_name: list[str] | dict[str, list[str]],
    ) -> pd.DataFrame:
        """
        因子正交 / 因子分组正交
        :param input_df: 因子数据
        :param factors_name: 因子名 -1 因子列表 -2 因子分组字典
        :return 对称正交后的因子
        """
        input_df_copy = input_df.copy(deep=True)

        if isinstance(factors_name, list):
            for date, group in input_df_copy.groupby("date"):
                group[factors_name] = processor.refactor.symmetric_orthogonal(group[factors_name])
        elif isinstance(factors_name, dict):
            for date, group in input_df_copy.groupby("date"):
                for group_factors in factors_name.values():
                    group[group_factors] = processor.refactor.symmetric_orthogonal(group[group_factors])
        else:
            raise TypeError

        return input_df_copy

    @staticmethod
    def factors_weighting(
            input_df: pd.DataFrame,
            factors_name: list[str] | dict[str, list[str]],
            factors_weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        因子加权
        :param input_df: 因子数据
        :param factors_name: 因子名 -1 因子列表 -2 因子分组字典
        :param factors_weights: 因子权重
        :return: 加权因子值
        """
        input_df_copy = input_df.copy(deep=True)

        # -1 计算加权因子
        if isinstance(factors_name, list):
            for date, group in input_df.groupby("date"):
                group[factors_name] = group[factors_name] * factors_weights.loc[date, factors_name]
                group[factors_name] = processor.dimensionless.standardization(group[factors_name])
        elif isinstance(factors_name, dict):
            for date, group in input_df_copy.groupby("date"):
                for group_factors in factors_name.values():
                    group[group_factors] = group[group_factors] * factors_weights.loc[date, group_factors]
                    group[group_factors] = processor.dimensionless.standardization(group[group_factors])
        else:
            raise TypeError

        return input_df_copy

    @staticmethod
    def synthesis_factor(
            input_df: pd.DataFrame,
            factors_synthesis_table: dict[str, list[str]],
            factors_weights: pd.DataFrame,
            keep_cols: list[str] = None
    ) -> pd.DataFrame:
        """
        因子合成
        :param input_df: 因子数据
        :param factors_synthesis_table: 因子合成字典
        :param factors_weights: 因子权重
        :param keep_cols: 需要保留的原始列名列表
        :return: 合成因子
        """
        result_dfs = []
        for date, group in input_df.copy(deep=True).groupby("date"):
            senior_dfs = {}

            # -1 计算合成因子
            for senior, component in factors_synthesis_table.items():
                senior_df = (group[component] * factors_weights.loc[date, component]).sum(axis=1, skipna=False)
                senior_dfs[senior] = senior_df

            # -2 构建当期合成因子df
            senior_dfs = pd.DataFrame(senior_dfs)

            # -3 添加需要保留的原始列（索引自动对齐）
            if keep_cols:
                for col in keep_cols:
                    if col in group.columns and col not in senior_dfs.columns:
                        senior_dfs[col] = group[col].values

            # -4 标准化（仅对因子列处理）
            factor_cols = list(factors_synthesis_table.keys())
            senior_dfs[factor_cols] = processor.dimensionless.standardization(senior_dfs[factor_cols])

            result_dfs.append(senior_dfs)

        return pd.concat(result_dfs).dropna(ignore_index=True)


###################################################
class PositionWeight:
    """仓位权重"""

    # --------------------------
    # 辅助方法
    # --------------------------
    @classmethod
    def __get_standardized_feature_ranking_vector(
            cls,
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
            date: processor.dimensionless.normalization(
                df[factor_name].rank(),
                feature_range=feature_range
            )
            for date, df in factors_data.items()
        }

    @classmethod
    def __get_group_standardized_feature_ranking_vector(
            cls,
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
                lambda x: processor.dimensionless.normalization(x, feature_range)
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
            factors_value: pd.DataFrame,
    ) -> pd.Series:
        """
        等权权重
        :param factors_value: 因子数据
        :return 仓位权重
        """
        result_dfs = []
        for date, group in factors_value.groupby("date"):
            result_dfs.append(
                pd.Series(
                    1 / group.shape[0],
                    index=group.index,
                    name="position_weight"
                )
            )
        print(pd.concat(result_dfs))
        print(dd)

        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    @classmethod
    def _get_group_equal_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.Series]:
        """
        等权权重（分组）
        :param factors_data: 因子数据
        """
        weight = {
            date: pd.Series(
                len(df["group"].unique()) / df.shape[0],
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
        对冲仓位权重（分组）
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param distribution 权重分布集中度
        :return: 对冲仓位权重
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
        纯多头仓位权重（分组）
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
            factors_value: pd.DataFrame,
            factor_name: str,
            method: POSITION_WEIGHT,
            distribution: tuple[float, float] = (1, 1)
    ) -> pd.Series:
        """
        获取仓位权重
        :param factors_value: 因子数据
        :param factor_name: 排序因子名
        :param method: 权重方法
        :param distribution 权重分布集中度
                p > 1	        强化头部集中度	            押注少数高得分标的
                p = 1	        线性权重（原样保留得分比例）	平衡策略
                0 < p < 1	    分散权重	                降低头部集中度，避免过度暴露
                p = 0	        等权重（需特殊处理）
        :return 仓位权重
        """
        cls._get_equal_weight(factors_value)
        print(dd)
        handlers = {
            "equal": lambda: cls._get_equal_weight(factors_value),
            "group_equal": lambda: cls._get_group_equal_weight(factors_value),
            "long_only": lambda: cls._get_long_only_weight(factors_value, factor_name, distribution),
            "group_long_only": lambda: cls._get_group_long_only_weight(factors_value, factor_name, distribution),
            "hedge": lambda: cls._get_hedge_weight(factors_value, factor_name, distribution),
            "group_hedge": lambda: cls._get_group_hedge_weight(factors_value, factor_name, distribution)
        }

        return handlers[method]()


###################################################
class FactorWeight:
    """因子权重"""

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
    @classmethod
    def _calc_equal_weight(
            cls,
            factors_value: pd.DataFrame,
            factors_name: list[str],
    ) -> pd.DataFrame:
        """
        等权权重
        :param factors_value: 因子数据
        :param factors_name: 因子名列表
        :return 等权权重
        """
        result_dfs = []
        for date, group in factors_value.groupby("date"):
            result_dfs.append(
                pd.Series(
                    1 / len(factors_name),
                    index=factors_name,
                    name=date
                )
            )

        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    @classmethod
    def _calc_ic_weight(
            cls,
            factors_value: pd.DataFrame,
            factors_name: list[str],
            window: int
    ) -> pd.DataFrame:
        """
        计算ic权重
        :param factors_value: 因子数据
        :param factors_name: 因子名列表
        :param window: 取均值滚动窗口数
        :return ic权重
        """
        # -1 rank ic
        rank_ic = cls.calc_rank_ic(factors_value, factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 滚动均值的绝对值 ic
        rolling_ic = rank_ic_shifted.rolling(window, min_periods=1).mean().abs()

        # -4 权重
        weights = rolling_ic.div(rolling_ic.sum(axis=1), axis="index")

        return weights

    @classmethod
    def _calc_ic_decay_weight(
            cls,
            factors_value: pd.DataFrame,
            factors_name: list[str],
            window: int
    ) -> pd.DataFrame:
        """
        计算ic指数加权权重
        :param factors_value: 因子数据
        :param factors_name: 因子名列表
        :param window: 取均值滚动窗口数
        :return ic权重
        """
        # -1 rank ic
        rank_ic = cls.calc_rank_ic(factors_value, factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 滚动指数加权均值的绝对值 ic
        rolling_ic = rank_ic_shifted.ewm(halflife=window, min_periods=1).mean().abs()

        # -4 权重
        weights = rolling_ic.div(rolling_ic.sum(axis=1), axis="index")

        return weights

    @classmethod
    def _calc_ir_weight(
            cls,
            factors_value: pd.DataFrame,
            factors_name: list[str],
            window: int
    ) -> pd.DataFrame:
        """
        计算ir权重
        :param factors_value: 因子数据
        :param factors_name: 因子名列表
        :param window: 滚动窗口数
        :return 因子权重
        """
        # -1 rank ic
        rank_ic = cls.calc_rank_ic(factors_value, factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 滚动均值的绝对值 ir
        rolling_ir = rank_ic_shifted.rolling(window, min_periods=1).apply(cls.calc_icir)

        # -4 权重
        weights = rolling_ir.div(rolling_ir.sum(axis=1), axis="index")

        return weights

    @classmethod
    def _calc_ir_decay_weight(
            cls,
            factors_value: pd.DataFrame,
            factors_name: list[str],
            window: int
    ) -> pd.DataFrame:
        """
        计算ir指数加权权重
        :param factors_value: 因子数据
        :param factors_name: 因子名列表
        :param window: 半衰期
        :return 因子权重
        """
        # -1 rank ic
        rank_ic = cls.calc_rank_ic(factors_value, factors_name)

        # -2 平移 rank ic
        rank_ic_shifted = rank_ic.shift(1)

        # -3 滚动均值的绝对值 ir
        ic_ewm_mean = rank_ic_shifted.ewm(halflife=window, min_periods=1).mean()
        ic_ewm_std = rank_ic_shifted.ewm(halflife=window, min_periods=1).std()
        rolling_ir = (ic_ewm_mean / ic_ewm_std).abs().replace(np.inf, 0).fillna(0)

        # -4 权重
        weights = rolling_ir.div(rolling_ir.sum(axis=1), axis="index")

        return weights

    @classmethod
    def _calc_ir_decay_weight_with_diff_halflife(
            cls,
            factors_value: pd.DataFrame,
            factors_name: list[str],
            half_life: pd.DataFrame
    ) -> pd.DataFrame:
        """
        不同半衰期的ir指数加权权重
        :param factors_value: 因子数据
        :param factors_name: 因子名列表
        :param half_life: 因子半衰期
        :return 因子权重
        """
        # -1 rank ic
        rank_ic = cls.calc_rank_ic(factors_value, factors_name)

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
            factors_value: pd.DataFrame,
            factors_name: list[str],
            method: FACTOR_WEIGHT,
            window: int = 12,
            half_life: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        获取因子权重
        :param factors_value: 因子数据
        :param factors_name: 因子名列表
        :param method: 计算因子权重方法
        :param window: 因子滚动均值窗口数
        :param half_life: 因子半衰期
        :return 因子权重
        """
        handlers = {
            "equal": lambda: cls._calc_equal_weight(
                factors_value,
                factors_name
            ),
            "ic_weight": lambda: cls._calc_ic_weight(
                factors_value,
                factors_name,
                window
            ),
            "ic_decay_weight": lambda: cls._calc_ic_decay_weight(
                factors_value,
                factors_name,
                window
            ),
            "ir_weight": lambda: cls._calc_ir_weight(
                factors_value,
                factors_name,
                window
            ),
            "ir_decay_weight": lambda: cls._calc_ir_decay_weight(
                factors_value,
                factors_name,
                window
            ),
            "_calc_ir_decay_weight_with_diff_halflife":
                lambda: cls._calc_ir_decay_weight_with_diff_halflife(
                    factors_value,
                    factors_name,
                    half_life
            ),
        }

        return handlers[method]()
