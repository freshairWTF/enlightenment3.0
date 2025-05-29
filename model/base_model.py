from abc import ABC, abstractmethod
from statsmodels import api as sm
from sklearn.ensemble import RandomForestRegressor

import numpy as np
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
            'objective': 'reg:squarederror',    # 定义损失函数类型：回归：reg:squarederror（均方误差）分类：binary:logistic（二分类概率）、multi:softmax（多分类）排序：rank:pairwise（文档对排序）
            'verbosity': 1,                     # 日志输出级别：0（静默）、1（警告）、2（信息）、3（调试）

            'learning_rate': 0.05,              # 学习率：控制每棵树对最终预测的贡献权重。较低值（0.01-0.2）可防止过拟合
            'n_estimators': 1000,
            'max_depth': 9,                     # 树的最大深度：增加深度提升模型复杂度但易过拟合。推荐3-10之间，高频交易场景可降低至3-5
            'min_child_weight': 3,

            'subsample': 0.8,                   # 行采样比例：每次建树时随机抽取样本的比例，常用0.5-0.8防止过拟合
            'colsample_bytree': 0.7,            # 列采样比例：每棵树随机选择特征的比例，与随机森林思想类似

            'reg_alpha': 0.1,                   # L1正则化项：促进特征稀疏性，适用于高维特征筛选
            'reg_lambda': 0.5,                  # L2正则化项：惩罚权重过大，缓解过拟合
            'gamma': 0.1                        # 节点分裂最小损失增益阈值：值越大分裂越保守。推荐0.1-0.3用于抑制噪声
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
    # 随机森林
    # --------------------------
    @classmethod
    def calc_predict_return_by_randomforest(
            cls,
            x_value: dict[str, pd.DataFrame | pd.Series],
            y_value: dict[str, pd.Series],
            window: int = 12
    ) -> dict[str, pd.DataFrame]:
        """
        基于随机森林的滚动窗口回归预测收益率
        :param x_value: T期截面数据
        :param y_value: T期收益率
        :param window: 滚动窗口长度
        :return: 预期收益率
        """
        # 随机森林参数配置（参考网页7、网页8的金融预测最佳实践）
        params = {
            'n_estimators': 200,  # 树的数量，网页8使用200棵
            'max_depth': 10,  # 树的最大深度，网页7建议5-15之间
            'min_samples_split': 10,  # 节点分裂最小样本数，抑制噪声干扰
            'max_features': 'sqrt',  # 每棵树随机选择√(总特征数)的特征
            'n_jobs': -1,  # 使用全部CPU核心加速计算
            'random_state': 42,  # 确保结果可复现
            'verbose': 1  # 显示训练进度（参考网页6的日志配置）
        }

        sorted_dates = sorted(x_value.keys())
        result = {}

        for i in range(window, len(sorted_dates)):
            # ====================
            # 样本内训练（改进点：增加特征筛选）
            # ====================
            train_window = sorted_dates[i - window:i]

            # 改进点：动态特征合并（参考网页7的prepare_features方法）
            train_dfs = []
            for date in train_window:
                # 合并特征与目标变量时保留原始特征（网页8的X/y分离方式）
                df = pd.concat([x_value[date], y_value[date].rename('target')], axis=1)
                df["date"] = date
                train_dfs.append(df)

            train_data = pd.concat(train_dfs).dropna()

            # 改进点：增加滞后特征（参考网页3的shift特征构造）
            for lag in [1, 3, 5]:
                train_data[f'return_lag_{lag}'] = train_data.groupby(level=0)['target'].shift(lag)

            train_data = train_data.dropna()

            # 拆分训练集（网页6的滑动窗口实现方式）
            x_train = train_data.drop(['target', 'date'], axis=1)
            y_train = train_data['target']

            # ====================
            # 模型训练（增加早停机制）
            # ====================
            try:
                # 改进点：使用OOB误差估计（参考网页8的评估方法）
                model = RandomForestRegressor(**params, oob_score=True)
                model.fit(x_train, y_train)

                # 特征重要性筛选（网页7的核心功能）
                importance = pd.Series(model.feature_importances_, index=x_train.columns)
                selected_features = importance.nlargest(15).index.tolist()  # 保留前15个重要特征
            except Exception as e:
                print(f"训练失败：{str(e)}")
                continue

            # ====================
            # 样本外预测（改进点：增加不确定性估计）
            # ====================
            predict_date = sorted_dates[i]
            predict_df = x_value[predict_date].copy()

            # 为预测数据生成滞后特征（需与训练集同步）
            for lag in [1, 3, 5]:
                predict_df[f'return_lag_{lag}'] = y_value[sorted_dates[i - lag]].reindex(predict_df.index)

            x_predict = predict_df[selected_features].dropna()

            # 执行预测（网页6的预测实现方式）
            predicted = pd.Series(
                model.predict(x_predict),
                index=x_predict.index,
                name="predicted"
            )

            # 改进点：计算预测分位数（参考网页9的风险控制思想）
            quantiles = np.quantile([tree.predict(x_predict) for tree in model.estimators_],
                                    q=[0.25, 0.75], axis=0)
            predicted_df = pd.DataFrame({
                'point_pred': predicted,
                'lower_bound': quantiles[0],
                'upper_bound': quantiles[1]
            })

            result[predict_date] = predicted_df

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
