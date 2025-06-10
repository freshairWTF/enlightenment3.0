"""因子降维"""

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

import pandas as pd
import statsmodels.api as sm

from utils.processor import DataProcessor


###################################################
class DimensionalityReduction:
    """因子降维"""

    processor = DataProcessor()

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

    @classmethod
    def factors_orthogonal(
            cls,
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

        result_dfs = []
        if isinstance(factors_name, list):
            for date, group in input_df_copy.groupby("date"):
                group[factors_name] = cls.processor.refactor.symmetric_orthogonal(group[factors_name])
                result_dfs.append(group)
        elif isinstance(factors_name, dict):
            for date, group in input_df_copy.groupby("date"):
                for group_factors in factors_name.values():
                    group[group_factors] = cls.processor.refactor.symmetric_orthogonal(group[group_factors])
                result_dfs.append(group)
        else:
            raise TypeError

        return pd.concat(result_dfs)

    @classmethod
    def synthesis_factor(
            cls,
            input_df: pd.DataFrame,
            factors_synthesis_table: dict[str, list[str]],
            factors_weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        因子合成
        :param input_df: 因子数据
        :param factors_synthesis_table: 因子合成字典
        :param factors_weights: 因子权重
        :return: 合成因子
        """
        result_dfs = []
        for date, group in input_df.copy().groupby("date"):
            senior_dfs = {}
            for senior, component in factors_synthesis_table.items():
                senior_df = (group[component] * factors_weights.loc[date, component]).sum(axis=1, skipna=False)
                senior_dfs[senior] = senior_df
            # 构建当期合成因子df
            senior_dfs = pd.DataFrame(senior_dfs)
            senior_dfs = cls.processor.dimensionless.standardization(senior_dfs)
            senior_dfs.insert(0, 'date', date)

            result_dfs.append(senior_dfs)

        return pd.concat(result_dfs).dropna(ignore_index=True)

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
