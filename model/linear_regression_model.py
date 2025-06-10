"""线性回归模型"""
import pandas as pd
from dataclasses import dataclass

from model_utils import FactorWeight
from utils.processor import DataProcessor
from model.model_utils import ModelUtils


########################################################################
class LinearRegressionModel:
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

        self.factors_setting = self.model_setting.factors_setting       # 因子设置
        self.factors_name = [                                           # 因子名
            f"processed_{f.factor_name}"
            for f in self.factors_setting
        ]
        self.processor = DataProcessor()                                # 数据处理
        self.utils = ModelUtils()                                       # 模型工具


        self.input_df = self._pre_processing(self.input_df, self.factors_setting)
        self._collinearity()

    def _pre_processing(
            self,
            raw_df: pd.DataFrame,
            factors_setting: list[dataclass]
    ) -> pd.DataFrame:
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

        result_dfs = []
        for date, group_df in raw_df.groupby("date"):
            try:
                processed_df = __process_single_date(group_df)
                result_dfs.append(processed_df)
            except ValueError:
                continue

        # 合并处理结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    def _collinearity(
            self
    ) -> pd.DataFrame:
        """
        因子降维
            -1 合成
            -2 降维
        """
        # -1 三级因子因子权重
        bottom_fw = FactorWeight(
            factors_value=self.input_df,
            factors_name=self.factors_name,
            method=self.model_setting.bottom_factor_weight_method
        )
        bottom_factors_weight = bottom_fw.get_factors_weights(12)

        DimensionalityReduction(self.model_setting).synthesis_factor(
            self.input_df, None, bottom_factors_weight)
        print(dd)


        # ---------------------------------------
        # 因子降维（去多重共线性 -> vif + 对称正交 + 预拟合）
        # ---------------------------------------
        self.logger.info("---------- 因子降维 ----------")
        # 因子降维
        collinearity = DimensionalityReduction(self.model_setting)
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

    @classmethod
    def calc_z_scores(
            cls,
            data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            weights: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        计算因子综合Z值
        :param data: 数据
        :param factors_name: T期因子名
        :param weights: 权重
        :return: 因子综合Z值
        """
        # -1 计算z值
        z_score = {
            date: filtered_df.rename("综合Z值").to_frame()
            for date, df in data.items()
            if not (
                filtered_df := (
                    df[factors_name[date]].mean(axis=1)
                ) if weights.empty
                else (
                        df[factors_name[date]] * weights.loc[date]
                ).sum(axis=1, skipna=True)
            ).dropna().empty
        }

        # -2 标准化
        z_score = {
            date: processed_df
            for date, df in z_score.items()
            if not (
                processed_df := cls.processor.dimensionless.standardization(df, error="ignore").dropna()
            ).empty
        }

        return z_score

    @classmethod
    def calc_predict_return(
            cls,
            x_value: dict[str, pd.DataFrame],
            y_value: dict[str, pd.Series],
            window: int = 12
    ) -> dict[str, pd.DataFrame]:
        """
        滚动窗口回归预测收益率
        :param x_value: T期截面数据
        :param y_value: T期收益率
        :param window: 滚动窗口长度
        :return: 预期收益率
        """
        # 按日期排序并转换为列表
        sorted_dates = sorted(x_value.keys())
        result = {}

        # 滚动窗口遍历
        for i in range(window, len(sorted_dates)):
            # ====================
            # 样本内训练
            # ====================
            # 获取训练窗口数据
            train_window = sorted_dates[i - window:i]

            # 合并窗口期数据
            train_dfs = []
            for date in train_window:
                df = pd.concat([x_value[date], y_value[date]], axis=1, join="inner")
                df["date"] = date
                train_dfs.append(df)
            train_data = pd.concat(train_dfs).dropna()

            # 准备训练数据
            x_train = sm.add_constant(train_data["综合Z值"], has_constant="add")
            y_train = train_data["pctChg"]

            # 训练模型
            try:
                model = sm.OLS(y_train, x_train).fit()
            except Exception as e:
                print(str(e))
                continue  # 处理奇异矩阵等异常情况

            # ====================
            # 样本外预测
            # ====================
            # 获取预测日数据
            predict_date = sorted_dates[i]
            predict_df = x_value[predict_date]

            # 生成预测特征
            x_predict = sm.add_constant(predict_df, has_constant="add")

            # 执行预测
            predicted = model.predict(x_predict)
            predicted.name = "predicted"

            # 存储结果
            result[predict_date] = predicted.to_frame()

        return result

    def run(self):
        """
        线性模型：
            1）计算因子权重；
            2）选择加权方法，计算综合Z-Score
            3）Z-Score回归/计算预期收益率；
            4）分组；
            5）仓位权重；
        """

        print(dd)


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
