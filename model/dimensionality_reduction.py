from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import statsmodels.api as sm
import scipy.cluster.hierarchy as sch


###################################################
class DimensionalityReduction:

    def __init__(self):
        pass


###################################################
import numpy as np
from sklearn.decomposition import PCA


class FactorPCA:
    """
    用PCA对因子矩阵进行降维的封装类

    参数:
    n_components - 主成分数量 (int/float/'auto')
    auto_threshold - 自动选择主成分时的累积方差阈值 (默认0.95)
    """

    def __init__(self, n_components='auto', auto_threshold=0.95):
        self.n_components = n_components
        self.auto_threshold = auto_threshold
        self.pca = None
        self.factor_names = None
        self.asset_index = None

    def _auto_components(self, variance_ratio):
        """自动选择满足方差阈值的主成分数量"""
        cum_var = np.cumsum(variance_ratio)
        return np.argmax(cum_var >= self.auto_threshold) + 1

    def fit(self, df):
        """
        拟合PCA模型
        输入:
        df - DataFrame, index为资产名称, columns为因子名称
        """
        # 保存数据标签
        self.factor_names = df.columns.tolist()
        self.asset_index = df.index

        # 自动确定主成分数量
        if self.n_components == 'auto':
            temp_pca = PCA()
            temp_pca.fit(df)
            self.n_components = self._auto_components(temp_pca.explained_variance_ratio_)

        # 正式拟合
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(df)
        return self

    def transform(self, df):
        """应用PCA转换"""
        if self.pca is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # 校验因子维度
        if list(df.columns) != self.factor_names:
            raise ValueError("因子名称与训练数据不一致")

        # PCA转换
        pca_result = self.pca.transform(df)

        # 转换为带标签的DataFrame
        return pd.DataFrame(
            pca_result,
            index=df.index,
            columns=[f'PC{i + 1}' for i in range(pca_result.shape[1])]
        )

    def fit_transform(self, df):
        """联合拟合和转换"""
        self.fit(df)
        return self.transform(df)

    def get_components(self):
        """获取主成分因子载荷"""
        return pd.DataFrame(
            self.pca.components_.T,
            index=self.factor_names,
            columns=[f'PC{i + 1}' for i in range(self.pca.n_components_)]
        )

    def get_variance(self):
        """获取方差解释信息"""
        return pd.DataFrame({
            '方差贡献率': self.pca.explained_variance_ratio_,
            '累积贡献率': np.cumsum(self.pca.explained_variance_ratio_)
        }, index=[f'PC{i + 1}' for i in range(self.pca.n_components_)])


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

        # 存储中间结果
        self.vif_removed_factors = {}  # 各日期被VIF剔除的因子
        self.cluster_groups = {}  # 各日期聚类分组结果

    def _auto_vif_reduction(
            self,
            df: pd.DataFrame
    ) -> list[str]:
        """
        基于VIF，逐步回归筛选因子
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
            df: pd.DataFrame,
            factors: list[str]
    ):
        """
        层次聚类降维
        :param df: 截面数据
        :param factors: 因子名
        :return:
        """
        corr_matrix = df[factors].corr().abs()
        distance_matrix = 1 - corr_matrix
        linkage = sch.linkage(distance_matrix, method='ward')
        clusters = sch.fcluster(linkage, t=self.cluster_threshold, criterion='distance')

        # 记录聚类分组
        cluster_dict = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(factors[i])

        # 选择代表因子
        selected_factors = []
        for cluster_id, cluster_factors in cluster_dict.items():
            # 选择组内平均相关性最小的因子（替代方案）
            avg_corr = corr_matrix.loc[cluster_factors, cluster_factors].mean(axis=1)
            best_factor = avg_corr.idxmin()
            selected_factors.append(best_factor)

        return selected_factors, cluster_dict

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
            selected_factors[date] = self._auto_vif_reduction(df[self.factors_name])

            if not selected_factors:
                continue

            # -------------------
            # -2 聚类降维筛选因子（非线性相关）
            # -------------------
            # final_factors, cluster_dict = self._hierarchical_clustering(df, selected_factors)
            # self.cluster_groups[date] = cluster_dict
            #
            # # 保存处理后的数据
            # processed_data[date] = df_clean[final_factors]

        return selected_factors

    def get_diagnostic_info(self):
        """ 获取处理过程的诊断信息 """
        return {
            "vif_removed": self.vif_removed_factors,
            "cluster_groups": self.cluster_groups
        }