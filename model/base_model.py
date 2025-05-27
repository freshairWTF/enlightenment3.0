from abc import ABC, abstractmethod
from statsmodels import api as sm

import pandas as pd
import xgboost as xgb

from data_processor import DataProcessor
from factor_weight import FactorWeight
from position_weight import PositionWeight


####################################################
class MultiFactorsModel(ABC):
    """多因子模型模板"""

    processor = DataProcessor
    factor_weight = FactorWeight
    position_weight = PositionWeight

    # --------------------------
    # 公开 API 方法
    # --------------------------
    @classmethod
    @abstractmethod
    def run(cls) -> dict[str, pd.DataFrame]:
        """
        线性模型：
            1）因子方向调整；
            2）因子/异象数据预处理；
            3）因子正交；
            4）多元线性回归加权；
            5）计算综合因子得分；
            6）综合因子得分一元回归；
            7）计算预期收益率
        """

    # --------------------------
    # 线性多因子回归
    # --------------------------
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
                processed_df := cls.processor.standardization(df, error="ignore").dropna()
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

    # --------------------------
    # xgboost多因子回归
    # --------------------------
    @classmethod
    def calc_weight_factors(
            cls,
            data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            weights: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        计算因子加权值
        :param data: 数据
        :param factors_name: T期因子名
        :param weights: 权重
        :return: 因子综合Z值
        """
        # -1 计算加权因子
        weight_factors = {
            date: filtered_df
            for date, df in data.items()
            if not (
                filtered_df := (
                        df[factors_name[date]] * weights.loc[date]
                )
            ).dropna(axis=1, how="all").dropna(how="any").empty
        }

        # -2 标准化
        return {
            date: cls.processor.standardization(df, error="ignore")
            for date, df in weight_factors.items()
        }

    @classmethod
    def calc_predict_return_by_xgboost(
            cls,
            x_value: dict[str, pd.DataFrame | pd.Series],
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
        # XGBoost参数配置
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
        }

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

            train_dfs = []
            for date in train_window:
                df = pd.concat([x_value[date], y_value[date]], axis=1, join="inner")
                df["date"] = date
                train_dfs.append(df)
            train_data = pd.concat(train_dfs).dropna()

            # 准备训练数据
            y_train = train_data["pctChg"]
            x_train = sm.add_constant(train_data.drop(["pctChg", "date"], axis=1), has_constant="add")

            try:
                model = xgb.XGBRegressor(**params).fit(
                    x_train,
                    y_train
                )
            except Exception as e:
                print(str(e))
                continue

            # ====================
            # 样本外预测
            # ====================
            # 获取预测日数据
            predict_date = sorted_dates[i]
            predict_df = x_value[predict_date]

            # 生成预测特征
            x_predict = sm.add_constant(predict_df, has_constant="add")
            # print(x_predict)

            # 执行预测
            predicted = pd.Series(
                model.predict(x_predict),
                index=x_predict.index
            )
            predicted.name = "predicted"

            # 存储结果
            result[predict_date] = predicted.to_frame()

        return result

    # --------------------------
    # 辅助方法
    # --------------------------
    @staticmethod
    def join_data(
            merge_data: dict[str, pd.DataFrame],
            merged_data: dict[str, pd.Series | pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        合并数据
        :param merge_data: 合并数据
        :param merged_data: 被合并数据
        :return: 合并数据后的数据
        """
        return {
            date: df.join(
                merged_data.get(date, pd.DataFrame()),
                how="left"
            )
            for date, df in merge_data.items()
        }
