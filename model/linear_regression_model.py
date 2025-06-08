"""线性回归模型"""
import pandas as pd
from dataclasses import dataclass

from utils.processor import DataProcessor
from model.dimensionality_reduction import FactorCollinearityProcessor


########################################################################
class LinearRegressionModel:
    """线性回归模型"""

    def __init__(
            self,
            input_df: pd.DataFrame,
            factors_setting: list[dataclass],
            individual_position_limit: float = 0.1,
            index_data: dict[str, pd.DataFrame] | None = None,
    ):
        """
        :param input_df: 数据
        :param factors_setting: 因子因子设置
        :param individual_position_limit: 单一持仓上限
        :param index_data: 指数数据
        """
        self.input_df = input_df
        self.factors_setting = factors_setting
        self.index_data = index_data
        self.individual_position_limit = individual_position_limit

        self.processor = DataProcessor()
        self._pre_processing(input_df, factors_setting)

    def _pre_processing(
            self,
            raw_df: pd.DataFrame,
            factors_setting: list[dataclass]
    ):
        """
        数据预处理
            -1 缩尾
            -2 标准化
            -3 中性化
            -4 缩尾
            -5 标准化
        :param raw_df: 初始数据
        :param factors_setting: 因子配置列表
        :return: 处理过的数据
        """
        def __process_single_date(
                df_: pd.DataFrame,
        ) -> pd.DataFrame:
            """单日数据处理"""
            for setting in factors_setting:
                factor_name = setting.factor_name
                processed_col = f"processed_{factor_name}"

                df_[processed_col] = df_[factor_name].copy()
                # -1 第一次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

                # -2 中性化
                if setting.market_value_neutral:
                    df_[processed_col] = self.processor.neutralization.market_value_neutral(
                        df_[processed_col],
                        df_["对数市值"],
                        winsorizer=self.processor.winsorizer.percentile,
                        dimensionless=self.processor.dimensionless.standardization
                    )
                if setting.industry_neutral:
                    df_[processed_col] = self.processor.neutralization.industry_neutral(
                        df_[processed_col],
                        df_["行业"]
                    )

                # -3 第二次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

            return df_

        print(raw_df)
        result_dfs = []
        for date, group_df in raw_df.groupby("date"):
            try:
                processed_df = __process_single_date(group_df)
                result_dfs.append(processed_df)
            except ValueError:
                continue

        print(pd.concat(result_dfs) if result_dfs else pd.DataFrame())
        print(dd)
        # 合并处理结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    def run(self):
        """
        线性模型：
            1）计算因子权重；
            2）选择加权方法，计算综合Z-Score
            3）Z-Score回归/计算预期收益率；
            4）分组；
            5）仓位权重；
        """
        # ---------------------------------------
        # 因子降维（去多重共线性 -> vif + 对称正交 + 预拟合）
        # ---------------------------------------
        self.logger.info("---------- 因子降维 ----------")
        # 因子降维
        collinearity = FactorCollinearityProcessor(self.model_setting)
        collinearity_data = collinearity.fit_transform(
            processed_data
        )
        selected_factors = {date: df.columns.tolist() for date, df in collinearity_data.items()}

        # 预拟合
        beta_feature = self.evaluate.test.calc_beta_feature(
            processed_data, processed_factors_name, "pctChg"
        )
        r_squared = self.evaluate.test.calc_r_squared(
            processed_data, processed_factors_name, "pctChg"
        )

        # -1 因子权重
        factor_weights = self.factor_weight.get_factors_weights(
            factors_data=self.input_df,
            factors_name=self.factors_name,
            method=self.factor_weight_method,
            window=self.factor_weight_window
        )

        # -2 综合Z值
        z_score = self.calc_z_scores(
            data=self.input_df,
            factors_name=self.factors_name,
            weights=factor_weights
        )

        # -3 预期收益率
        predict_return = self.calc_predict_return(
            x_value=z_score,
            y_value={date: df["pctChg"] for date, df in self.input_df.items()},
            window=self.factor_weight_window
        )

        # -4 分组
        grouped_data = QuantProcessor.divide_into_group(
            predict_return,
            factor_col="",
            processed_factor_col="predicted",
            group_mode=self.group_mode,
            group_nums=self.group_nums,
            group_label=self.group_label,
        )

        # -5 仓位权重
        position_weight = self.position_weight.get_weights(
            grouped_data,
            factor_name="predicted",
            method=self.position_weight_method,
            distribution=self.position_distribution
        )

        # -6 数据合并
        result = self.join_data(grouped_data, position_weight)
        result = self.join_data(result, z_score)
        result = self.join_data(result, self.input_df)

        return result
