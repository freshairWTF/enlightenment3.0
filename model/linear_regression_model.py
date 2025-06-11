"""线性回归模型"""
import pandas as pd

from dataclasses import dataclass
from itertools import chain

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

        # self.utils.factor_weight.get_factors_weights(
        #     factors_value=input_df,
        #     factors_name=self.factors_name,
        #     method=self.model_setting.bottom_factor_weight_method,
        #     window=self.model_setting.factor_weight_window
        # )

    def _pre_processing(
            self,
            input_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        数据预处理
            -1 缩尾
            -2 标准化
            -3 中性化
            -4 缩尾
            -5 标准化
        :param input_df: 初始数据
        :return: 处理过的数据
        """
        def __process_single_date(
                df_: pd.DataFrame,
        ) -> pd.DataFrame:
            """单日数据处理"""
            for setting in self.factors_setting:
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
        for date, group_df in input_df.groupby("date"):
            try:
                processed_df = __process_single_date(group_df)
                result_dfs.append(processed_df)
            except ValueError:
                continue

        # 合并处理结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    def _bottom_factors_synthesis(
            self,
            input_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        三级因子合成二级因子
        :param input_df: 初始数据
        """
        # -1 三级因子构造表
        factors_synthesis_table = self.utils.extract.get_factors_synthesis_table(
            self.factors_setting,
            top_level=False
        )

        # -2 三级因子因子权重
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

        # -3 二级因子合成
        return self.utils.synthesis.synthesis_factor(
            input_df=input_df,
            factors_synthesis_table=factors_synthesis_table,
            factors_weights=factors_weight,
            keep_cols=["股票代码", "行业", "pctChg"]
        )

    def _level_2_factors_synthesis(
            self,
            input_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        二级因子合成一级因子
        :param input_df: 初始数据
        """
        # -1 二级因子构造表
        factors_synthesis_table = self.utils.extract.get_factors_synthesis_table(self.factors_setting)

        # -2 二级因子因子权重
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

        # -3 一级因子合成
        return self.utils.synthesis.synthesis_factor(
            input_df=input_df,
            factors_synthesis_table=factors_synthesis_table,
            factors_weights=factors_weight
        )

    def _comprehensive_z_value_synthesis(
            self,
            input_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        一级因子合成综合Z值
        :param input_df: 初始数据
        """
        # -1 综合Z值构造表
        factors_synthesis_table = {
            "综合Z值": list(
                chain.from_iterable(
                    self.utils.extract.get_factors_synthesis_table(self.factors_setting).values()
                )
            )
        }

        # -2 一级因子因子权重
        factors_weight = self.utils.factor_weight.get_factors_weights(
            factors_value=input_df,
            factors_name=factors_synthesis_table["综合Z值"],
            method=self.model_setting.bottom_factor_weight_method,
            window=self.model_setting.factor_weight_window
        )

        # -3 一级因子合成
        return self.utils.synthesis.synthesis_factor(
            input_df=input_df,
            factors_synthesis_table=factors_synthesis_table,
            factors_weights=factors_weight
        )

    @classmethod
    def model_training_and_predict(
            cls,
            x_value: dict[str, pd.DataFrame],
            y_value: dict[str, pd.Series],
            window: int = 12
    ) -> dict[str, pd.DataFrame]:
        """
        模型训练与预测
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
        线性模型处理流程：
            -1 因子数值处理
            -2 三级因子降维 -> 二级因子
            -3 二级因子降维 -> 综合Z值
            -4 模型训练、预测
            -5 收益率预测分组
            -6 仓位权重配比
        """
        self.input_df = self._pre_processing(self.input_df)
        print(self.input_df)
        print(self.input_df.columns)

        level_2_df = self._bottom_factors_synthesis(self.input_df)
        print(level_2_df)
        print(level_2_df.columns)

        level_2_df = pd.concat(
            [
                self._bottom_factors_synthesis(self.input_df),
                self.input_df[["date", "股票代码", "pctChg"]]
            ],
            axis=1,
            keys=["date", "股票代码"]
        )
        print(level_2_df)
        print(level_2_df.columns)
        level_1_df = pd.concat(
            [
                self._level_2_factors_synthesis(level_2_df),
                self.input_df[["date", "股票代码", "pctChg"]]
            ],
            axis=1
        )
        print(level_1_df)
        comprehensive_z_value = self._comprehensive_z_value_synthesis(level_1_df)
        print(comprehensive_z_value)
        print(dd)

        # -3 预期收益率
        predict_return = self.model_training_and_predict(
            x_value=z_score,
            y_value={date: df["pctChg"] for date, df in self.input_df.items()},
            window=self.factor_weight_window
        )

        # -4 分组
        grouped_data = self.processor.classification.divide_into_group(
            predict_return,
            factor_col="",
            processed_factor_col="predicted",
            group_mode=self.model_setting.group_mode,
            group_nums=self.model_setting.group_nums,
            group_label=self.model_setting.group_label,
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
