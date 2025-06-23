"""线性回归模型"""
from dataclasses import dataclass
from type_ import Literal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd

from utils.processor import DataProcessor
from model.model_utils import ModelUtils
from model.optimizer import PortfolioOptimizer


########################################################################
class LinearRegressionTestModel:
    """线性回归模型"""

    def __init__(
            self,
            input_df: pd.DataFrame,
            model_setting: dataclass,
            individual_position_limit: float = 0.1,
            index_data: dict[str, pd.DataFrame] | None = None,
    ):
        """
        :param input_df: 数据
        :param model_setting: 模型设置
        :param individual_position_limit: 单一持仓上限
        :param index_data: 指数数据
        """
        self.input_df = input_df
        self.model_setting = model_setting
        self.index_data = index_data
        self.individual_position_limit = individual_position_limit

        self.factors_setting = self.model_setting.factors_setting   # 因子设置
        self.processor = DataProcessor()                            # 数据处理
        self.utils = ModelUtils()                                   # 模型工具

        # 保留列
        self.keep_cols = ["date", "股票代码", "行业", "pctChg", "市值", "close", "volume"]

    def _direction_reverse(
            self,
            input_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        因子方向反转（历史回测）
        """
        reverse_df = input_df.copy(deep=True)

        for setting in self.factors_setting:
            if setting.reverse:
                reverse_df[setting.factor_name] *= -1

        return reverse_df

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

                # -1 正态变换
                if setting.transfer:
                    df_[processed_col] = self.processor.refactor.yeo_johnson_transfer(df_[processed_col])

                # -2 第一次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

                # -3 中性化
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

                # -4 第二次 去极值、标准化
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

    def _factors_weighting(
            self,
            input_df: pd.DataFrame,
            prefix: str = "processed"
    ) -> pd.DataFrame:
        """
        因子 -> 加权因子
        :param input_df: 初始数据
        :param prefix: 因子前缀
        :return 加权因子
        """
        # -1 三级因子 ic
        bottom_factors_name = [f"{prefix}_{f.factor_name}" for f in self.factors_setting]

        # -2 三级因子因子权重
        factors_weight = self.utils.factor_weight.get_factors_weights(
            factors_value=input_df,
            factors_name=bottom_factors_name,
            method=self.model_setting.bottom_factor_weight_method,
            window=self.model_setting.factor_weight_window
        )

        # -3 加权因子
        return self.utils.synthesis.factors_weighting(
            input_df,
            bottom_factors_name,
            factors_weight
        )

    def _factors_synthesis(
            self,
            input_df: pd.DataFrame,
            synthesis_table: dict[str, list[str]] | None = None,
            mode: Literal["THREE_TO_TWO", "TWO_TO_ONE", "ONE_TO_Z", "TWO_TO_Z", "THREE_TO_Z"] | None = None
    ) -> pd.DataFrame:
        """
        三级因子合成二级因子
            -1 三级因子 -> 二级因子
            -2 二级因子 -> 一级因子
            -3 一级因子 -> 综合Z值
            -4 二级因子 -> 综合Z值
            -5 三级因子 -> 综合Z值
        :param input_df: 初始数据
        :param mode: 因子生成模式
        """
        # -1 因子构造表
        if mode:
            factors_synthesis_table = self.utils.extract.get_factors_synthesis_table(
                self.factors_setting,
                mode=mode,
                prefix="processed"
            )
        elif synthesis_table:
            factors_synthesis_table = synthesis_table
        else:
            raise TypeError(f"输入有效参数: {synthesis_table} | {mode}")

        # -2 因子权重
        factors_weight = []
        for group_factors in factors_synthesis_table.values():
            factors_weight.append(
                self.utils.factor_weight.get_factors_weights(
                    factors_value=input_df,
                    factors_name=group_factors,
                    method=self.model_setting.bottom_factor_weight_method,
                    window=self.model_setting.factor_weight_window
                )
            )
        factors_weight = pd.concat(factors_weight, axis=1)

        # -3 因子合成
        return self.utils.synthesis.synthesis_factor(
            input_df=input_df,
            factors_synthesis_table=factors_synthesis_table,
            factors_weights=factors_weight,
            keep_cols=self.keep_cols
        )

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
                sorted_dates[i - window: i+1]), ["date", "股票代码", "close", "行业", "volume"]
            ]
            price_df = portfolio_df.pivot(
                index="date",
                columns="股票代码",
                values="close"
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
            # 模型评估
            # ====================
            metrics_series = self.calculate_metrics(y_true, true_df["predict"])
            metrics_series.name = predict_date
            metrics.append(metrics_series)

            # ====================
            # 数据整合（原值、预测收益率/分组、仓位权重、实际股数）
            # ====================
            result_dfs.append(true_df)

        return (pd.concat(result_dfs, ignore_index=True),
                pd.concat(metrics, axis=1).mean(axis=1).to_frame(name="value").T)

    def calculate_metrics(
            self,
            y_true: pd.Series,
            y_pred: pd.Series
    ) -> pd.Series:
        """计算评估指标"""
        metrics = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
        }
        return pd.Series(metrics)

    def portfolio_optimizer(
            self,
            price_df: pd.DataFrame,
            volume_df: pd.DataFrame,
            industry_df: pd.DataFrame,
            individual_upper: float = 1.0,
            individual_lower: float = 0.0,
            industry_upper: float = 1.0,
            industry_lower: float = 0.0
    ) -> pd.Series:
        """
        资产组合优化
        :param price_df: 资产价格df（分组）
        :param volume_df: 成交量df（分组）
        :param industry_df: 行业映射df（分组）
        :param individual_upper: 个体配置上限
        :param individual_lower: 个体配置下限
        :param industry_upper: 行业配置上限
        :param industry_lower: 行业配置下限
        """
        portfolio = PortfolioOptimizer(
            asset_prices=price_df,
            volume=volume_df,
            cycle=self.model_setting.cycle,
            cov_method="ledoit_wolf",
            shrinkage_target="constant_variance"
        )
        weights = portfolio.optimize_weights(
            objective="max_sharpe",
            weight_bounds=(individual_lower, individual_upper),
            sector_mapper=industry_df.to_dict(),
            sector_lower={ind: industry_lower for ind in industry_df.values},
            sector_upper={ind: industry_upper for ind in industry_df.values},
            clean=True,
        )

        return weights

    def run(
            self
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        线性模型处理流程：
            -1 因子数值处理
            -2 因子降维
            -3 模型训练、预测
            -4 收益率预测分组
            -5 仓位权重配比
        """
        # ----------------------------------
        # 数值处理
        # ----------------------------------
        # -1 方向反转
        self.input_df = self._direction_reverse(self.input_df)
        # -2 预处理
        self.input_df = self._pre_processing(self.input_df)
        # -3 对称正交
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

        # -3 合成 综合Z值
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

        return pred_df, estimate_metric
