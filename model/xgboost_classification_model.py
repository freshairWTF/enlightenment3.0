"""xgboost分类模型"""
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

import shap
import pandas as pd

from template import ModelTemplate


########################################################################
class XGBoostClassificationModel(ModelTemplate):
    """XGBoost分类模型"""

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

        # 分类器参数
        self.keep_cols.append("group")
        self.le = LabelEncoder()                                                # 标签转换实例

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
            # -1 获取训练窗口数据
            train_window = sorted_dates[i - window: i]
            # -2 获取训练数据
            x_train, y_train = (
                input_df.loc[input_df["date"].isin(train_window), x_cols],
                input_df.loc[input_df["date"].isin(train_window), y_col]
            )
            # -4 标签转换 离散字符 -> 连续整数
            y_train = pd.Series(
                self.le.fit_transform(y_train),
                index=y_train.index
            )

            # -5 训练模型
            model = XGBClassifier(objective='multi:softmax')
            model.fit(
                x_train,
                y_train,
            )

            # ====================
            # 样本外预测
            # ====================
            # -1 获取预测日数据
            predict_date = sorted_dates[i]
            # -2 获取测试窗口数据
            true_df = input_df.loc[input_df["date"] == predict_date]
            x_true, y_true = (
                input_df.loc[input_df["date"] == predict_date, x_cols],
                input_df.loc[input_df["date"] == predict_date, y_col]
            )
            # -3 模型预测
            true_df["predict"] = model.predict(x_true)
            # -4 标签转换 连续整数 -> 离散字符
            true_df["predict"] = pd.Series(
                self.le.inverse_transform(true_df["predict"]),
                index=true_df.index
            )
            print(true_df)
            print(true_df.columns)
            # ====================
            # 归因分析
            # ====================
            explainer = shap.TreeExplainer(model)
            print(explainer.shap_values(x_true))
            print([f"shap_{col}" for col in x_cols])
            print(x_true.index)
            shap_df = pd.DataFrame(
                explainer.shap_values(x_true),
                columns=[f"shap_{col}" for col in x_cols],
                index=x_true.index  # 保持与原始数据索引一致
            )
            true_df = pd.concat([true_df, shap_df], axis=1)

            # ====================
            # 数据整合（原值、预测收益率/分组、仓位权重、实际股数）
            # ====================
            result_dfs.append(true_df)

            # ====================
            # 模型评估
            # ====================
            try:
                metrics_series = self.calculate_classification_metrics(y_true, true_df["predict"])
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
        # 数据预处理
        self.input_df = self._pre_processing(self.input_df)
        # 标签预设
        self.input_df = self.processor.classification.divide_into_group(
            self.input_df,
            processed_factor_col="pctChg",
            group_mode=self.model_setting.group_mode,
            group_nums=self.model_setting.group_nums,
            group_label=self.model_setting.group_label
        )


        # ----------------------------------
        # 因子降维
        # ----------------------------------
        level_2_df = self._factors_synthesis(
            self.input_df,
            mode="THREE_TO_TWO"
        )

        # ----------------------------------
        # 因子相关性
        # ----------------------------------
        corr_df = self.calculate_factors_corr(
            factors_df=level_2_df,
            factors_name=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist()
        )

        # ----------------------------------
        # 模型
        # ----------------------------------
        pred_df, estimate_metric = self.model_training_and_predict(
            input_df=level_2_df,
            x_cols=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist(),
            y_col="group",
            window=self.model_setting.factor_weight_window
        )

        return {
            "模型": pred_df,
            "模型评估": estimate_metric,
            "因子相关性": corr_df,
            "因子shap值": pred_df.filter(like='shap_').abs().mean().sort_values(ascending=False)
        }
