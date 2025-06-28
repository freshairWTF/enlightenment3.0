"""
线性回归模型
    -1 等频分组
    -2 大于0后的等频分组
"""
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression

import pandas as pd

from template import ModelTemplate


########################################################################
class FightLinearRegressionModel(ModelTemplate):
    """线性回归模型（等频分组）"""

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

                # -1 方向变化
                if setting.reverse:
                    df_[processed_col] *= -1

                # -2 正态变换
                if setting.transfer:
                    df_[processed_col] = self.processor.refactor.yeo_johnson_transfer(df_[processed_col])

                # -3 第一次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

                # -4 中性化
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

                # -5 第二次 去极值、标准化
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

    def model_training_and_predict(
            self,
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

        # 评估指标
        metrics = []

        # 滚动窗口遍历
        result_dfs = []
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
            true_df = input_df.loc[input_df["date"] == predict_date]
            x_true, y_true = (
                input_df.loc[input_df["date"] == predict_date, x_cols],
                input_df.loc[input_df["date"] == predict_date, y_col]
            )
            # 模型预测
            true_df["predict"] = model.predict(x_true)

            # ====================
            # 预测分组
            # ====================
            true_df["group"] = self.processor.classification.frequency(
                true_df,
                factor_col="",
                processed_factor_col="predict",
                group_nums=self.model_setting.group_nums,
                group_label=self.model_setting.group_label,
                negative=False
            )

            # ====================
            # 权重优化（仓位权重、实际股数）
            # ====================
            # -1 数据转换
            portfolio_df = input_df.loc[input_df["date"].isin(
                sorted_dates[i - window: i+1]),
                ["date", "股票代码", "close", "行业", "volume", "pctChg"]
            ]
            price_df = portfolio_df.pivot(
                index="date",
                columns="股票代码",
                values="pctChg"
            )
            volume_df = portfolio_df.pivot(
                index="date",
                columns="股票代码",
                values="volume"
            )
            industry_df = portfolio_df.set_index("股票代码")["行业"]
            # -2 权重（分组）
            weights_series = []
            for _, group_df in true_df.groupby("group"):
                weights = self.portfolio_optimizer(
                    price_df=price_df[group_df["股票代码"].tolist()].ffill().bfill(),
                    volume_df=volume_df[group_df["股票代码"].tolist()],
                    industry_df=industry_df[industry_df.index.isin(group_df["股票代码"].tolist())]
                )
                weights_series.append(weights)
            # -3 合并
            true_df = pd.merge(
                true_df,
                pd.concat(weights_series).rename('position_weight'),
                left_on='股票代码',
                right_index=True,
                how='left'
            )

            # ====================
            # 数据整合（原值、预测收益率/分组、仓位权重、实际股数）
            # ====================
            result_dfs.append(true_df)

            # ====================
            # 模型评估
            # ====================
            try:
                metrics_series = self.calculate_regression_metrics(y_true, true_df["predict"])
                metrics_series.name = predict_date
                metrics.append(metrics_series)
            except ValueError:
                continue

        return (pd.concat(result_dfs, ignore_index=True),
                pd.concat(metrics, axis=1).mean(axis=1).to_frame(name="value").T)

    def run(
            self
    ) -> dict[str, pd.DataFrame]:
        """
        线性模型处理流程：
            -1 因子数值处理
            -2 因子升维/降维
            -3 模型训练、预测、分组
        """
        # ----------------------------------
        # 数值处理
        # ----------------------------------
        # -1 预处理
        self.input_df = self._pre_processing(self.input_df)
        # -2 对称正交
        self.input_df = self.utils.feature.factors_orthogonal(
            self.input_df,
            factors_name=self.utils.extract.get_factors_synthesis_table(
                self.factors_setting,
                mode="THREE_TO_TWO"
            )
        )

        # ----------------------------------
        # 因子降维
        # ----------------------------------
        # -1 因子合成
        level_2_df = self._factors_synthesis(self.input_df, mode="THREE_TO_TWO")
        # -2 pca降维
        pca_df = self.utils.feature.pca(
            level_2_df,
            factors_synthesis_table=self.utils.extract.get_factors_synthesis_table(
                self.factors_setting,
                mode="TWO_TO_ONE"
            ),
            keep_cols=self.keep_cols
        )

        # ----------------------------------
        # 因子相关性
        # ----------------------------------
        corr_df = self.calculate_factors_corr(
            factors_df=pca_df,
            factors_name=pca_df.columns[~pca_df.columns.isin(self.keep_cols)].tolist()
        )

        # ----------------------------------
        # 合成 综合Z值
        # ----------------------------------
        comprehensive_z_df = self._factors_synthesis(
            pca_df,
            synthesis_table={"综合Z值": pca_df.columns[~pca_df.columns.isin(self.keep_cols)].tolist()}
        )

        # ----------------------------------
        # 模型
        # ----------------------------------
        pred_df, estimate_metric = self.model_training_and_predict(
            input_df=comprehensive_z_df,
            x_cols=["综合Z值"],
            y_col="pctChg",
            window=self.model_setting.factor_weight_window
        )

        return {
            "模型": pred_df,
            "模型评估": estimate_metric,
            "因子相关性": corr_df
        }


########################################################################
class FightLinearRegressionHigherModel(ModelTemplate):
    """线性回归模型（高于0等频分组）"""

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

                # -1 方向变化
                if setting.reverse:
                    df_[processed_col] *= -1

                # -2 正态变换
                if setting.transfer:
                    df_[processed_col] = self.processor.refactor.yeo_johnson_transfer(df_[processed_col])

                # -3 第一次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

                # -4 中性化
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

                # -5 第二次 去极值、标准化
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

    def model_training_and_predict(
            self,
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

        # 评估指标
        metrics = []

        # 滚动窗口遍历
        result_dfs = []
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
            true_df = input_df.loc[input_df["date"] == predict_date]
            x_true, y_true = (
                input_df.loc[input_df["date"] == predict_date, x_cols],
                input_df.loc[input_df["date"] == predict_date, y_col]
            )
            # 模型预测
            true_df["predict"] = model.predict(x_true)

            # ====================
            # 预测分组
            # ====================
            true_df["group"] = self.processor.classification.frequency(
                true_df,
                factor_col="",
                processed_factor_col="predict",
                group_nums=self.model_setting.group_nums,
                group_label=self.model_setting.group_label,
                negative=False
            )

            # ====================
            # 总仓位计算（若加上总仓位限制，容易得不到最优解，整体缩放更易实现）
            # ====================
            """
            市场波动率 -> 总仓位？悦海
            指数预测 -> 总仓位
            个股 -> 整体缩放  跟最优化有区别吗？
            """
            positive_ratio = (true_df["predict"] > 0).sum() / true_df["predict"].count()

            # ====================
            # 权重优化（仓位权重、实际股数）
            # ====================
            # -1 数据转换
            portfolio_df = input_df.loc[input_df["date"].isin(
                sorted_dates[i - window: i+1]),
                ["date", "股票代码", "close", "行业", "volume", "pctChg"]
            ]
            price_df = portfolio_df.pivot(
                index="date",
                columns="股票代码",
                values="pctChg"
            )
            volume_df = portfolio_df.pivot(
                index="date",
                columns="股票代码",
                values="volume"
            )
            industry_df = portfolio_df.set_index("股票代码")["行业"]
            # -2 权重（分组）
            weights_series = []
            for _, group_df in true_df.groupby("group"):
                weights = self.portfolio_optimizer(
                    price_df=price_df[group_df["股票代码"].tolist()].ffill().bfill(),
                    volume_df=volume_df[group_df["股票代码"].tolist()],
                    industry_df=industry_df[industry_df.index.isin(group_df["股票代码"].tolist())]
                )
                weights_series.append(weights)
            # -3 合并
            true_df = pd.merge(
                true_df,
                pd.concat(weights_series).rename('position_weight'),
                left_on='股票代码',
                right_index=True,
                how='left'
            )

            # ====================
            # 数据整合（原值、预测收益率/分组、仓位权重、实际股数）
            # ====================
            result_dfs.append(true_df)

            # ====================
            # 模型评估
            # ====================
            try:
                metrics_series = self.calculate_regression_metrics(y_true, true_df["predict"])
                metrics_series.name = predict_date
                metrics.append(metrics_series)
            except ValueError:
                continue

        return (pd.concat(result_dfs, ignore_index=True),
                pd.concat(metrics, axis=1).mean(axis=1).to_frame(name="value").T)

    def run(
            self
    ) -> dict[str, pd.DataFrame]:
        """
        线性模型处理流程：
            -1 因子数值处理
            -2 因子升维/降维
            -3 模型训练、预测、分组
        """
        # ----------------------------------
        # 数值处理
        # ----------------------------------
        # -1 预处理
        self.input_df = self._pre_processing(self.input_df)
        # -2 对称正交
        self.input_df = self.utils.feature.factors_orthogonal(
            self.input_df,
            factors_name=self.utils.extract.get_factors_synthesis_table(
                self.factors_setting,
                mode="THREE_TO_TWO"
            )
        )

        # ----------------------------------
        # 因子降维
        # ----------------------------------
        # -1 因子合成
        level_2_df = self._factors_synthesis(self.input_df, mode="THREE_TO_TWO")
        # -2 pca降维
        pca_df = self.utils.feature.pca(
            level_2_df,
            factors_synthesis_table=self.utils.extract.get_factors_synthesis_table(
                self.factors_setting,
                mode="TWO_TO_ONE"
            ),
            keep_cols=self.keep_cols
        )

        # ----------------------------------
        # 因子相关性
        # ----------------------------------
        corr_df = self.calculate_factors_corr(
            factors_df=pca_df,
            factors_name=pca_df.columns[~pca_df.columns.isin(self.keep_cols)].tolist()
        )

        # ----------------------------------
        # 合成 综合Z值
        # ----------------------------------
        comprehensive_z_df = self._factors_synthesis(
            pca_df,
            synthesis_table={"综合Z值": pca_df.columns[~pca_df.columns.isin(self.keep_cols)].tolist()}
        )

        # ----------------------------------
        # 模型
        # ----------------------------------
        pred_df, estimate_metric = self.model_training_and_predict(
            input_df=comprehensive_z_df,
            x_cols=["综合Z值"],
            y_col="pctChg",
            window=self.model_setting.factor_weight_window
        )

        return {
            "模型": pred_df,
            "模型评估": estimate_metric,
            "因子相关性": corr_df
        }

