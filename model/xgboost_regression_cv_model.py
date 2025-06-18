"""xgboost回归模型"""
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from type_ import Literal

import numpy as np
import pandas as pd

from utils.processor import DataProcessor
from model.model_utils import ModelUtils


########################################################################
class XGBoostRegressionCVModel:
    """XGBoost回归模型（超参寻优）"""

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

        self.factors_setting = self.model_setting.factors_setting  # 因子设置
        self.processor = DataProcessor()  # 数据处理
        self.utils = ModelUtils()  # 模型工具

        self.keep_cols = ["date", "股票代码", "行业", "pctChg", "市值"]  # 保留列

        # 超参数网格
        # n_estimators/learning_rate
        # max_depth/min_child_weight/gamma
        # reg_alpha/reg_lambda
        # subsample/colsample_bytree
        self.model_param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [10, 50, 100, 500, 1000, 3000, 5000]
        }

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

            # 3.1 时间序列交叉验证 + 超参数优化
            tscv = TimeSeriesSplit(n_splits=5, gap=1)
            model = XGBRegressor(objective='reg:squarederror')

            # 网格搜索寻优
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.model_param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(
                x_train,
                y_train,
                # eval_set=[(x_valid, y_valid)],
                # early_stopping_rounds=100,
            )
            best_model = grid_search.best_estimator_

            # 提取每次CV的最优参数
            results = pd.DataFrame(grid_search.cv_results_)
            best_params_per_fold = []
            for grid_i in range(grid_search.n_splits_):
                # 筛选当前折的最佳参数组合（按排名rank=1）
                best_idx = results[f'split{grid_i}_test_score'].idxmax()
                best_params = results.loc[best_idx, 'params']
                best_score = results.loc[best_idx, f'split{grid_i}_test_score']
                best_params_per_fold.append((best_params, best_score))
                print(f"Fold {grid_i + 1} - Best Params: {best_params}, Score: {-best_score:.4f}")

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
                data=best_model.predict(x_test),
                index=x_test.index,
                name="predict"
            )
            output_df = input_df.loc[input_df["date"] == predict_date]
            output_df["predict"] = y_pred

            print(predict_date)
            print(output_df["date"].unique())
            result_dfs.append(output_df)

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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        线性模型处理流程：
            -1 因子数值处理
            -2 因子降维
            -3 模型训练、预测
            -4 收益率预测分组
            -5 仓位权重配比
        """
        # -1 数据处理
        self.input_df = self._direction_reverse(self.input_df)
        self.input_df = self._pre_processing(self.input_df)

        # -2 因子合成
        level_2_df = self._factors_synthesis(
            self.input_df,
            mode="THREE_TO_TWO"
        )

        # -3 模型训练、预测
        pred_df, estimate_metric = self.model_training_and_predict(
            input_df=level_2_df,
            x_cols=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist(),
            y_col="pctChg",
            window=self.model_setting.factor_weight_window
        )
        print(pred_df['date'].unique())

        # -4 模型后续处理
        classification_df = self.processor.classification.divide_into_group(
            pred_df,
            factor_col="",
            processed_factor_col="predict",
            group_mode=self.model_setting.group_mode,
            group_nums=self.model_setting.group_nums,
            group_label=self.model_setting.group_label,
        )

        # -5 仓位权重
        position_weight = self.utils.pos_weight.get_weights(
            classification_df,
            factor_col="predict",
            method=self.model_setting.position_weight_method,
            distribution=self.model_setting.position_distribution
        )

        return position_weight, estimate_metric
