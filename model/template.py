"""模型模板"""

from dataclasses import dataclass
from type_ import Literal
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    f1_score, matthews_corrcoef, precision_score, recall_score
)
from sklearn.linear_model import LinearRegression

import shap
import numpy as np
import pandas as pd

from utils.processor import DataProcessor
from model.model_utils import ModelUtils
from model.optimizer import PortfolioOptimizer

"""
组内权重的再管理
# -5 仓位权重
position_weight = self.utils.pos_weight.get_weights(
    classification_df,
    factor_col="predict",
    method=self.model_setting.position_weight_method,
    distribution=self.model_setting.position_distribution
)
"""

########################################################################
class ModelTemplate:
    """模型模板"""

    def __init__(
            self,
            input_df: pd.DataFrame,
            model_setting: dataclass,
            descriptive_factors: list[str]
    ):
        """
        :param input_df: 数据
        :param model_setting: 模型设置
        :param descriptive_factors: 描述性统计因子
        """
        self.input_df = input_df
        self.model_setting = model_setting

        self.factors_setting = self.model_setting.factors_setting       # 因子设置
        self.total_capital = self.model_setting.total_capital           # 策略分配资金
        self.processor = DataProcessor()                                # 数据处理
        self.utils = ModelUtils()                                       # 模型工具

        # 保留列
        self.keep_cols = [
            "date", "股票代码", "行业", "pctChg", "市值",
            "close", "volume", "origin_group"
        ]
        self.keep_cols += descriptive_factors
        self.keep_cols = list(set(self.keep_cols))

    # ------------------------------------------
    # 因子降维
    # ------------------------------------------
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

    # ------------------------------------------
    # 最优化
    # ------------------------------------------
    def portfolio_optimizer(
            self,
            price_df: pd.DataFrame,
            volume_df: pd.DataFrame,
            industry_df: pd.DataFrame,
            individual_upper: float = 1.0,
            individual_lower: float = 0.0,
            industry_upper: float = 1.0,
            industry_lower: float = 0.0,
            allocation: bool = False,
            last_price_series: pd.Series | None = None
    ) -> tuple[pd.Series, pd.Series]:
        """
        资产组合权重优化
        :param price_df: 资产价格df（分组）
        :param volume_df: 成交量df（分组）
        :param industry_df: 行业映射df（分组）
        :param individual_upper: 个体配置上限
        :param individual_lower: 个体配置下限
        :param industry_upper: 行业配置上限
        :param industry_lower: 行业配置下限
        :param allocation: 计算具体股数
        :param last_price_series: 最新股价序列（用于计算股数）
        :return 个股权重
        """
        portfolio = PortfolioOptimizer(
            asset_prices=price_df,
            volume=volume_df,
            cycle=self.model_setting.cycle,
            cov_method="ledoit_wolf",
            shrinkage_target="constant_variance"
        )
        weights = portfolio.optimize_weights(
            objective="min_volatility",
            weight_bounds=(individual_lower, individual_upper),
            sector_mapper=industry_df.to_dict(),
            sector_lower={ind: industry_lower for ind in industry_df.values},
            sector_upper={ind: industry_upper for ind in industry_df.values},
            clean=True,
        )
        if allocation:
            alloc = portfolio.discrete_allocation(
                last_price=last_price_series,
                total_portfolio_value=self.total_capital,
            )
        else:
            alloc = pd.Series()

        return weights, alloc

    # ------------------------------------------
    # 评估方法
    # ------------------------------------------
    @classmethod
    def calculate_regression_metrics(
            cls,
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

    def calculate_classification_metrics(
            self,
            y_true: pd.Series,
            y_pred: pd.Series
    ) -> pd.Series:
        """计算评估指标"""
        metrics = {
            "f1_score": f1_score(y_true, y_pred, average="macro"),
            "mcc": matthews_corrcoef(y_true, y_pred),
            **{f"precision_{k}": v for k, v in
               zip(self.model_setting.group_label, precision_score(y_true, y_pred, average=None))},
            **{f"recall_{k}": v for k, v in
               zip(self.model_setting.group_label, recall_score(y_true, y_pred, average=None))},
        }

        return pd.Series(metrics)

    @staticmethod
    def shap_for_multiclass(
            model,
            factors_name: list[str],
            x_true: pd.Series,
            y_true: list[str],
            y_predict: list[str],
    ) -> pd.DataFrame:
        """
        基于shap的归因分析（多分类模型）
        :param model: 模型
        :param factors_name: 因子名
        :param x_true: 待解释数据集
        :param y_true: 真实标签
        :param y_predict: 预测标签
        :return 预测、真实的shap值，以及二者差值
        """
        # -1 解释器
        explainer = shap.TreeExplainer(model)
        # -2 SHAP值（[样本数，特征数，类别]）
        shap_values = explainer.shap_values(x_true)
        # -3 预测类别的SHAP值
        pred_shap = shap_values[np.arange(len(x_true)), :, y_predict]
        # -4 真实类别的SHAP值
        true_shap = shap_values[np.arange(len(x_true)), :, y_true]
        # -5 差异分析（定位认知偏差）
        delta_shap = pred_shap - true_shap

        # -6 数据整合
        shap_pred_df = pd.DataFrame(
            pred_shap,
            columns=[f'{col}_pred' for col in factors_name],
            index=x_true.index
        )
        shap_true_df = pd.DataFrame(
            true_shap,
            columns=[f'{col}_true' for col in factors_name],
            index=x_true.index
        )
        shap_delta_df = pd.DataFrame(
            delta_shap,
            columns=[f'{col}_delta' for col in factors_name],
            index=x_true.index
        )

        # 合并最终结果
        return pd.concat(
            [shap_pred_df, shap_true_df, shap_delta_df],
            axis=1
        )

    @staticmethod
    def shape_for_linear(
            model,
            factors_name: list[str],
            x_train: pd.DataFrame,
            x_true: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        基于shap的归因分析（多分类模型）
        :param model: 模型
        :param factors_name: 因子名
        :param x_train: 训练集
        :param x_true: 测试集
        :return 预测shap值
        """
        # -1 解释器
        explainer = shap.LinearExplainer(model, x_train)
        # -2 SHAP值（[样本数，特征数，类别]）
        shap_values = explainer.shap_values(x_true)
        # -3 预测类别的SHAP值（即模型预测值的解释）
        pred_shap = shap_values
        # -4 生成df
        shap_df = pd.DataFrame(
            pred_shap,
            columns=[f"{col}_shap" for col in factors_name],
            index=x_true.index
        )
        # -5 因子值与shap值合并
        result_df = pd.concat([shap_df, x_true], axis=1)
        # -6 移除列名前缀
        result_df.columns = result_df.columns.str.replace("processed_", '', regex=False)

        return result_df

    def calculate_linear_importance(
            self,
            model_metrics: pd.Series,
            x_cols: list[str],
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_true: pd.DataFrame,
            y_true: pd.Series
    ) -> pd.DataFrame:
        """
        线性因子重要性检查（单因子剔除：减少该因子计算评估指标的差值）
        :param model_metrics: 模型评估指标
        :param x_cols: 因子列名
        :param x_train: 训练因子集
        :param y_train: 训练目标集
        :param x_true: 真实因子集
        :param y_true: 真实目标集
        :return:
        """
        metrics = []
        for feature in x_cols:

            # -1 剔除该因子后训练、预测模型
            reduced_cols = [col for col in x_cols if col != feature]
            reduced_model = LinearRegression()
            reduced_model.fit(x_train[reduced_cols], y_train)
            reduced_pred = reduced_model.predict(x_true[reduced_cols])

            # -2 计算模型评估指标变化
            reduced_metrics = self.calculate_regression_metrics(y_true, reduced_pred)
            metrics.append(
                {
                    "因子": feature,
                    "delta_MAE": reduced_metrics['MAE'] - model_metrics['MAE'],
                    "delta_RMSE": reduced_metrics['RMSE'] - model_metrics['RMSE'],
                    "delta_R2": reduced_metrics['R2'] - model_metrics['R2'],
                }
            )

        return pd.DataFrame(metrics)

    # ------------------------------------------
    # 其他
    # ------------------------------------------
    def _direction_reverse(
            self,
            input_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        因子方向反转（权重为负值）
        """
        reverse_df = input_df.copy(deep=True)

        # -1 三级因子 ic
        bottom_factors_name = [f.factor_name for f in self.factors_setting]
        factors_ic = (
            self.utils.factor_weight.calc_rank_ic(reverse_df, bottom_factors_name)
            .shift(1)
            .rolling(12, min_periods=1).mean()
        )

        # -2 因子值反转（依据因子滚动IC均值）
        for factor_name in bottom_factors_name:
            negative_dates = factors_ic[factors_ic[factor_name] < 0].index
            reverse_df.loc[reverse_df["date"].isin(negative_dates), factor_name] *= -1

        return reverse_df

    def calculate_factors_corr(
            self,
            factors_df: pd.DataFrame,
            factors_name: list[str] | None = None,
            mode: Literal["ONE_TO_Z", "TWO_TO_Z", "THREE_TO_Z"] | None = None
    ) -> pd.DataFrame:
        """
        计算因子间相关性
        :param factors_df: 因子数据
        :param factors_name: 因子名列表
        :param mode: 因子生成模式
        :return: 因子间相关性
        """
        # -1 因子构造表
        if mode:
            factors_name = self.utils.extract.get_factors_synthesis_table(
                self.factors_setting,
                mode=mode,
                prefix="processed"
            )["综合Z值"]
        elif factors_name:
            factors_name = factors_name
        else:
            raise TypeError(f"输入有效参数: {factors_name} | {mode}")

        return factors_df[factors_name].corr()
