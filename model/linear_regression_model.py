"""线性回归模型"""
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd

from template import ModelTemplate
"""
    -1 多项式 是/否
        是 -> 仅生成交叉项 是/否
    -2 因子合成 是/否 
        -> 是：二级因子/综合Z值
            -> 二级因子 特征提取 是/否
        -> 否：三级因子 特征提取 是/否
    -3 线性模型
        基础线性模型
        lasso
        
"""


########################################################################
class LinearRegressionModel(ModelTemplate):
    """线性回归模型"""

    def __init__(
            self,
            input_df: pd.DataFrame,
            model_setting: dataclass,
            descriptive_factors: list[str],
            index_data: dict[str, pd.DataFrame] | None = None,
    ):
        """
        :param input_df: 数据
        :param model_setting: 模型设置
        :param descriptive_factors: 描述性统计因子
        :param index_data: 指数数据
        """
        super().__init__(
            input_df,
            model_setting,
            descriptive_factors
        )
        self.index_data = index_data

    def _pre_processing(
            self,
            input_df: pd.DataFrame,
            prefix: str = "processed"
    ) -> pd.DataFrame:
        """
        数据预处理
            -1 缩尾
            -2 标准化
            -3 中性化
            -4 缩尾
            -5 标准化
        :param input_df: 初始数据
        :param prefix: 预处理生成因子前缀
        :return: 处理过的数据
        """
        def __process_single_date(
                df_: pd.DataFrame,
        ) -> pd.DataFrame:
            """单日数据处理"""
            for setting in self.factors_setting:
                factor_name = setting.factor_name
                processed_col = f"{prefix}_{factor_name}"

                df_[processed_col] = df_[factor_name].copy()
                # -1 第一次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

                # -2 中性化
                if setting.market_value_neutral:
                    df_[processed_col] = self.processor.neutralization.log_market_cap(
                        df_[processed_col],
                        df_["对数市值"],
                        winsorizer=self.processor.winsorizer.percentile,
                        dimensionless=self.processor.dimensionless.standardization
                    )
                if setting.industry_neutral:
                    df_[processed_col] = self.processor.neutralization.industry(
                        df_[processed_col],
                        df_["行业"]
                    )

                # -3 第二次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

            return df_

        result_dfs = []
        for date, group_df in input_df.groupby("date"):
            try:
                processed_df = __process_single_date(group_df)
                result_dfs.append(processed_df)
            except ValueError:
                continue

        # 合并处理结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    @classmethod
    def model_training_and_predict(
            cls,
            input_df: pd.DataFrame,
            x_cols: list[str],
            y_col: str,
            window: int = 12
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        模型训练与预测
        :param input_df: 输入数据
        :param x_cols: 因子列名
        :param y_col: 目标列名
        :param window: 滚动窗口长度
        :return: 预期收益率
        """
        # 按日期排序并转换为列表
        sorted_dates = sorted(input_df["date"].unique())

        result_dfs = []
        metrics = {
            'MAE': [],
            'RMSE': [],
            'R2': []
        }
        # 滚动窗口遍历
        for i in range(window, len(sorted_dates)):
            # ====================
            # 样本内训练
            # ====================
            # 获取训练窗口数据
            train_window = sorted_dates[i - window: i]
            x_train, y_train = (
                input_df.loc[input_df["date"].isin(train_window), x_cols],
                input_df.loc[input_df["date"].isin(train_window), y_col]
            )
            # 训练模型
            try:
                model = LinearRegression(fit_intercept=True)
                model.fit(x_train, y_train)
            except Exception as e:
                print(str(e))
                continue

            # ====================
            # 样本外预测
            # ====================
            # 获取预测日数据
            predict_date = sorted_dates[i]
            # 获取测试窗口数据
            x_test, y_test = (
                input_df.loc[input_df["date"] == predict_date, x_cols],
                input_df.loc[input_df["date"] == predict_date, y_col]
            )
            # 模型预测
            y_pred = pd.Series(
                data=model.predict(x_test),
                index=x_test.index,
                name="predict"
            )
            result_dfs.append(
                pd.concat(
                    [
                        input_df.loc[input_df["date"] == predict_date],
                        y_pred
                    ],
                    axis=1)
            )
            # ====================
            # 模型评估
            # ====================
            metrics['MAE'].append(mean_absolute_error(y_test, y_pred))
            metrics['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics['R2'].append(r2_score(y_test, y_pred))

        # ====================
        # 模型评估指标聚合
        # ====================
        metrics = pd.DataFrame(
            {
                k: np.nanmean(v) if v else np.nan
                for k, v in metrics.items()
            },
            index=["value"]
        )

        return pd.concat(result_dfs, ignore_index=True), metrics

    def run(
            self
    ) -> dict[str, pd.DataFrame]:
        """
        线性模型处理流程：
            -1 因子数值处理
            -2 因子升维/降维
            -3 模型训练、预测、分组
        """
        # -1 数据处理
        self.input_df = self._direction_reverse(self.input_df)
        self.input_df = self._pre_processing(self.input_df)
        # self.input_df = self.utils.feature.create_polynomial(self.input_df, interaction_only=False)

        # -2 因子合成
        # level_2_df = self._factors_synthesis(self.input_df, mode="THREE_TO_TWO")
        # level_1_df = self._factors_synthesis(level_2_df, mode="TWO_TO_ONE")
        # comprehensive_z_df = self._factors_synthesis(level_1_df, mode="ONE_TO_Z")

        # weighting_factors_df = self._factors_weighting(self.input_df)
        # comprehensive_z_df = self._factors_synthesis(weighting_factors_df, mode="THREE_TO_Z")
        self.input_df = self.utils.feature.factors_orthogonal(
            self.input_df,
            self.utils.extract.get_factors_synthesis_table(
                self.factors_setting,
                mode="THREE_TO_TWO"
            )
        )

        level_2_df = self._factors_synthesis(self.input_df, mode="THREE_TO_TWO")

        level_2_df = self.utils.feature.pca(
            level_2_df,
            self.utils.extract.get_factors_synthesis_table(
                self.factors_setting,
                mode="TWO_TO_ONE"
            ),
            keep_cols=self.keep_cols
        )

        comprehensive_z_df = self._factors_synthesis(
            level_2_df,
            synthesis_table={"综合Z值": level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist()}
        )

        # comprehensive_z_df = self._factors_synthesis(level_2_df, mode="TWO_TO_Z")

        # -3 模型训练、预测
        pred_df, estimate_metric = self.model_training_and_predict(
            input_df=comprehensive_z_df,
            x_cols=["综合Z值"],
            y_col="pctChg",
            window=self.model_setting.factor_weight_window
        )

        return {
            "模型": pred_df,
            "模型评估": estimate_metric,
            "因子相关性": pd.DataFrame()
        }
