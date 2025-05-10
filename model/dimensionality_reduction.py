"""因子降维"""

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

import pandas as pd
import statsmodels.api as sm
import scipy.cluster.hierarchy as sch

from constant.type_ import FACTOR_WEIGHT
from data_processor import DataProcessor
from factor_weight import FactorWeight


###################################################
class FactorCollinearityProcessor:
    """
    处理多日期因子数据共线性问题
    """

    def __init__(
            self,
            primary_factors: dict[str, list[str]],
            secondary_factors: dict[str, list[str]],
            weight_method: FACTOR_WEIGHT,
            window: int,
            vif_threshold: float = 10,
            cluster_threshold: float = 0.7
    ):
        """
        :param primary_factors: 一级行业分类
        :param secondary_factors: 二级行业分类
        :param weight_method: 加权方法
        :param window: 窗口数
        :param vif_threshold: VIF筛选阈值，高于此值的因子被剔除
        :param cluster_threshold: 聚类距离阈值（0-1之间，值越小聚类越细）
        """
        self.primary_factors = primary_factors
        self.secondary_factors = secondary_factors

        self.weight_method = weight_method
        self.window = window
        self.vif_threshold = vif_threshold
        self.cluster_threshold = cluster_threshold

    def _auto_vif_reduction(
            self,
            df: pd.DataFrame
    ) -> list[str]:
        """
        基于VIF，逐步回归筛选因子
        终止条件1：仅剩截距项时退出循环（len(variables) <= 1）。
        终止条件2：剩余变量的VIF最大值低于阈值（vif.max() < self.vif_threshold）。
        :param 截面数据
        :return 无多重共线性的因子
        """
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
            if vif.max() < self.vif_threshold:
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

    def _hierarchical_clustering(
            self,
            df: pd.DataFrame
    ) -> list[str]:
        """
        层次聚类降维
        :param df: vif筛选后的截面数据
        :return: 层次聚类后的因子
        """
        if df.empty:
            return []
        if df.shape[1] == 1:
            return df.columns.tolist()

        factors_name = df.columns.tolist()

        # 计算相关系数矩阵
        corr_matrix = df.corr().abs()
        # 构建距离矩阵
        distance_matrix = 1 - corr_matrix

        # 层次聚类
        linkage = sch.linkage(
            distance_matrix,
            method='ward'
        )
        clusters = sch.fcluster(
            linkage,
            t=self.cluster_threshold,
            criterion='distance'
        )

        # 记录聚类分组
        cluster_dict = {}
        for idx, cluster_id in enumerate(clusters):
            factor_name = factors_name[idx]
            cluster_dict.setdefault(cluster_id, []).append(factor_name)

        # 选择代表因子
        selected_factors = []
        for cluster_id, cluster_factors in cluster_dict.items():
            # 选择组内平均相关性最小的因子
            avg_corr = corr_matrix.loc[cluster_factors, cluster_factors].mean(axis=1)
            best_factor = avg_corr.idxmin()
            selected_factors.append(best_factor)

        return selected_factors

    def _bottom_factors_orthogonal(
            self,
            processed_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        三级底层因子正交
        :param processed_data: 预处理因子数据
        :return 对称正交后的因子
        """
        for date, df in processed_data.items():
            for second, factors in self.secondary_factors.items():
                processed_data[date][factors] = DataProcessor.symmetric_orthogonal(df[factors])
        return processed_data

    def _synthesis_secondary_factor(
            self,
            processed_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        合成二级因子
        :param processed_data: 对称正交后的因子数据
        :return 合成的二级因子
        """
        result = pd.DataFrame()
        fw = FactorWeight
        for second, secondary_factors in self.secondary_factors.items():

            # -1 因子权重
            factors_weights = fw.get_factors_weights(
                processed_data,
                {date: secondary_factors for date in processed_data.keys()},
                self.weight_method,
                self.window
            )

            # -2 合成综合因子
            second_df = fw.synthesis_factor(
                processed_data,
                {date: secondary_factors for date in processed_data.keys()},
                factors_weights,
                second
            ).dropna()

            # -3 数据合并
            if result.empty:
                result = second_df
            else:
                result = second_df.merge(result, on=['index', 'date'], how='inner')

        return result

    def _pca(
            self,
            secondary_factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        PCA降维
        :return:
        """
        result = []
        for date, df in secondary_factors.groupby("date"):

            # ------------------------------
            # 截面PCA降维
            # ------------------------------
            section_df = pd.DataFrame()
            for factors in self.primary_factors.values():
                # -1 二级因子分类PCA降维
                pca = PCA(n_components=0.95)
                pca_result = pd.DataFrame(
                    pca.fit_transform(df[factors]),
                    index=df.index,
                    columns=[f'PC{i + 1}' for i in range(pca.n_components_)]
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
        return pd.concat(
            result,
            axis=0,
            ignore_index=True
        )

    def fit_transform(
            self,
            processed_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        处理全部日期数据
        :param processed_data: 输入数据 {date: df}
        :return: 降维后的数据
        """
        # T期滚动 全部
        # 单次 多次

        # -1 三级底层因子正交
        processed_data = self._bottom_factors_orthogonal(processed_data)

        # -2 二级因子合成
        secondary_factors_df = self._synthesis_secondary_factor(
            processed_data
        )

        # -3 二级因子降维
        pca_df = self._pca(secondary_factors_df)

        # -4 转换为 {date: pd.DataFrame}格式
        result = {
            str(date): group.set_index('index').drop("date", axis=1)
            for date, group in pca_df.groupby('date')
        }

        return result
