"""因子降维"""

from collections import defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

import pandas as pd
import statsmodels.api as sm
import scipy.cluster.hierarchy as sch

from constant.quant_setting import ModelSetting
from utils.processor import DataProcessor
from model.factor_weight import FactorWeight


###################################################
class FactorCollinearityProcessor:
    """
    处理多日期因子数据共线性问题
    """

    def __init__(
            self,
            model_setting: ModelSetting,
            vif_threshold: float = 10,
            cluster_threshold: float = 0.7
    ):
        """
        :param vif_threshold: VIF筛选阈值，高于此值的因子被剔除
        :param cluster_threshold: 聚类距离阈值（0-1之间，值越小聚类越细）
        """
        self.model_setting = model_setting
        self.vif_threshold = vif_threshold
        self.cluster_threshold = cluster_threshold

        self.primary_factors = self._get_primary_factors()              # 一级行业分类
        self.secondary_factors = self._get_secondary_factors()          # 二级行业分类

    # ------------------------------------------
    # 初始化方法
    # ------------------------------------------
    def _get_primary_factors(
            self
    ) -> dict[str, list[str]]:
        """获取一级分类因子"""
        result = defaultdict(list)
        for setting in self.model_setting.factors_setting:
            result[setting.primary_classification].append(setting.secondary_classification)
        return {k: list(dict.fromkeys(v)) for k, v in result.items()}

    def _get_secondary_factors(
            self
    ) -> dict[str, list[str]]:
        """获取二级分类因子"""
        result = defaultdict(list)
        for setting in self.model_setting.factors_setting:
            if "_sqr" in setting.factor_name:
                result[setting.secondary_classification].append(setting.factor_name)
            else:
                result[setting.secondary_classification].append(f"processed_{setting.factor_name}")
        return {k: list(dict.fromkeys(v)) for k, v in result.items()}

    def _get_half_life(
            self
    ) -> pd.DataFrame:
        """获取因子半衰期"""
        result = {}
        for setting in self.model_setting.factors_setting:
            result[setting.factor_name]= [setting.half_life]
        return pd.DataFrame(result, index=["half_life"]).add_prefix("processed_")

    # ------------------------------------------
    # 降维方法
    # ------------------------------------------
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
                processed_data[date][factors] = DataProcessor().refactor.symmetric_orthogonal(df[factors])
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
        weight_method = self.model_setting.bottom_factor_weight_method

        for second, secondary_factors in self.secondary_factors.items():

            # -1 因子权重
            factors_weights = fw.get_factors_weights(
                processed_data,
                {date: secondary_factors for date in processed_data.keys()},
                weight_method,
                self.model_setting.factor_weight_window,
                self._get_half_life()
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
            for primary, second_factors in self.primary_factors.items():
                # -1 二级因子分类PCA降维
                pca = PCA(n_components=0.95)
                pca_result = pd.DataFrame(
                    pca.fit_transform(df[second_factors]),
                    index=df.index,
                    columns=[f"{primary}{i+1}" for i in range(pca.n_components_)]
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
    # ------------------------------------------
    # 公开 API方法
    # ------------------------------------------
    def fit_transform(
            self,
            processed_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        处理全部日期数据
        :param processed_data: 输入数据 {date: df}
        :return: 降维后的数据
        """
        """
        1、因子回测 需要检测出 因子的非线性特征
                  标记 尽管ic不过关，但是icir大于0.5的弱因子，用以合成增强因子（在因子回测时加入因子合成模块）
        2、因子合成 u型因子加入二项式
        3、模型 非线性模型 主要是xgboost与随机森林
        """
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
            str(date): group.set_index('index').drop("date", axis=1).dropna(axis=1, how="all").dropna(how="any")
            for date, group in pca_df.groupby('date')
        }

        return result
