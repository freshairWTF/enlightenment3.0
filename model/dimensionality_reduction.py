"""因子降维"""

from collections import defaultdict

from baostock.demo.demo_TradeDates import result
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

import pandas as pd
import statsmodels.api as sm
import scipy.cluster.hierarchy as sch

from constant.quant_setting import ModelSetting
from utils.processor import DataProcessor


###################################################
class DimensionalityReduction:
    """因子降维"""

    processor = DataProcessor()

    def __init__(
            self,
            model_setting: ModelSetting,
    ):
        """
        :param model_setting: 模型设置参数
        """
        self.model_setting = model_setting

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

    def factors_orthogonal(
            self,
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
                group[factors_name] = self.processor.refactor.symmetric_orthogonal(group[factors_name])
                result_dfs.append(group)
        elif isinstance(factors_name, dict):
            for date, group in input_df_copy.groupby("date"):
                for group_factors in factors_name.values():
                    group[group_factors] = self.processor.refactor.symmetric_orthogonal(group[group_factors])
                result_dfs.append(group)
        else:
            raise TypeError

        return pd.concat(result_dfs)

    def synthesis_factor(
            self,
            input_df: pd.DataFrame,
            factors_synthesis_table: dict[str, list[str]],
            factors_weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        因子合成
        :param input_df: 因子数据
        :param factors_synthesis_table: 因子合成表
        :param factors_weights: 因子权重
        :return: 合成因子
        """

        """
                    close        date  ...  processed_换手率标准差_2  processed_close
0        8.190000  2024-04-12  ...            0.110629        -0.455488
1       13.500000  2024-04-12  ...            2.103793        -0.032258
2        3.870000  2024-04-12  ...            0.848214        -0.799811
3        4.550000  2024-04-12  ...           -0.575908        -0.745612
4        2.400000  2024-04-12  ...           -0.331888        -0.916976
...           ...         ...  ...                 ...              ...
256955  46.360001  2100-01-01  ...           -0.294561         1.777494
256956  38.599998  2100-01-01  ...           -0.725595         1.289488
256957  26.910000  2100-01-01  ...           -0.910297         0.554335
256958  82.089996  2100-01-01  ...            0.500080         3.444639
256959  65.620003  2100-01-01  ...           -1.219432         2.988705

[256960 rows x 150 columns]
{'估值因子': ['processed_对数市值', 'processed_对数市值_rolling_normalized', 'processed_市值', 'processed_市值_rolling_normalized', 'processed_市现率倒数', 'processed_市销率倒数', 'processed_市销率倒数_rolling_normalized', 'processed_市净率倒数', 'processed_实际收益率', 'processed_核心利润盈利市值比', 'processed_周期市盈率倒数'], '规模因子': ['processed_负债和所有者权益', 'processed_净经营资产', 'processed_所有者权益', 'processed_存货', 'processed_非流动资产合计', 'processed_金融性负债', 'processed_金融性资产', 'processed_经营性流动负债', 'processed_经营性流动资产', 'processed_经营性营运资本', 'processed_经营性长期负债', 'processed_经营性长期资产', 'processed_净负债', 'processed_净经营性长期资产', 'processed_少数股东权益', 'processed_实收资本'], '质量因子': ['processed_经营净利润', 'processed_净利润', 'processed_归属于母公司所有者的净利润', 'processed_利润总额', 'processed_营业利润', 'processed_营业收入', 'processed_毛利', 'processed_息税前利润', 'processed_核心利润'], '现金流量因子': ['processed_分配股利、利润或偿付利息所支付的现金', 'processed_购买商品、接受劳务支付的现金', 'processed_经营活动产生的现金流量净额', 'processed_投资活动现金流出小计', 'processed_投资活动现金流入小计', 'processed_销售商品、提供劳务收到的现金', 'processed_支付的其他与经营活动有关的现金', 'processed_支付给职工以及为职工支付的现金', 'processed_偿还债务支付的现金', 'processed_筹资活动现金流出小计', 'processed_筹资活动现金流入小计', 'processed_购建固定资产、无形资产和其他长期资产支付的现金', 'processed_经营活动现金流出小计', 'processed_经营活动现金流入小计', 'processed_取得借款收到的现金', 'processed_取得投资收益所收到的现金', 'processed_收到的其他与经营活动有关的现金', 'processed_收到的其他与投资活动有关的现金', 'processed_收到的税费返还', 'processed_支付的各项税费', 'processed_支付的其他与投资活动有关的现金', 'processed_投资活动产生的现金流量净额'], '动量因子': ['processed_累加收益率_0.09', 'processed_累加收益率_0.17', 'processed_累加收益率_0.25'], '流动性因子': ['processed_换手率均线_0.25', 'processed_换手率均线_0.5', 'processed_换手率均线_1', 'processed_换手率均线_1.5'], '流动性风险因子': ['processed_换手率标准差_0.25', 'processed_换手率标准差_0.5', 'processed_换手率标准差_1', 'processed_换手率标准差_1.5', 'processed_换手率标准差_2'], '行为金融因子': ['processed_close']}
            processed_对数市值  ...  processed_close
2024-04-12             NaN  ...              NaN
2024-04-19             NaN  ...              NaN
2024-04-26        0.010618  ...         0.004374
2024-04-30        0.010944  ...         0.027123
2024-05-10        0.006255  ...         0.012165
...                    ...  ...              ...
2025-05-09        0.010709  ...         0.004993
2025-05-16        0.012128  ...         0.006068
2025-05-23        0.011955  ...         0.006619
2025-05-30        0.011238  ...         0.006823
2100-01-01        0.011703  ...         0.007627

[61 rows x 71 columns]
        """
        print(input_df)
        print(self.secondary_factors)
        print(factors_weights)
        print(dd)

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


    @classmethod
    def _synthesis_factor(
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
