"""因子降维"""

from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import statsmodels.api as sm
import scipy.cluster.hierarchy as sch


###################################################
class FactorCollinearityProcessor:
    """
    处理多日期因子数据共线性问题（先VIF后聚类降维）
    """

    def __init__(
            self,
            factors_name: list[str],
            vif_threshold: float = 10,
            cluster_threshold: float = 0.7
    ):
        """
        :param factors_name: 因子名
        :param vif_threshold: VIF筛选阈值，高于此值的因子被剔除
        :param cluster_threshold: 聚类距离阈值（0-1之间，值越小聚类越细）
        """
        self.factors_name = factors_name
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
        variables = ["const"] + self.factors_name

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

    def fit_transform(
            self,
            processed_data: dict[str, pd.DataFrame],
    ) -> dict[str, list[str]]:
        """
        处理全部日期数据
        :param processed_data: 输入数据 {date: df}
        :return: 处理后的数据 {date: df_selected}
        """
        selected_factors = {}
        for date, df in processed_data.items():
            # -------------------
            # -1 VIF筛选因子（线性相关）
            # -------------------
            vif_factors = self._auto_vif_reduction(df[self.factors_name])

            if not vif_factors:
                continue

            # -------------------
            # -2 聚类降维筛选因子（非线性相关）
            # -------------------
            cluster_factors = self._hierarchical_clustering(df[vif_factors])

            # -------------------
            # -3 保存处理后的数据
            # -------------------
            selected_factors[date] = cluster_factors

        return selected_factors
