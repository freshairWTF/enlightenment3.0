from abc import ABC, abstractmethod
from scipy.stats import spearmanr
from statsmodels import api as sm

import numpy as np
import pandas as pd

from data_processor import DataProcessor
from constant.type_ import FACTOR_WEIGHT, validate_literal_params


####################################################
class FactorWeight:
    """因子权重"""

    # --------------------------
    # 辅助方法
    # --------------------------
    @staticmethod
    def _calc_ir(
            ic: pd.Series
    ) -> float:
        """
        计算ir
        :param ic: ic值
        :return ir值
        """
        ic_clean = ic.dropna()
        if len(ic_clean) < 2:
            return 0.0
        return np.abs(ic_clean.mean() / ic_clean.std()) if ic_clean.std() != 0 else 0.0

    @staticmethod
    def __calc_rank_ic(
            df: pd.DataFrame,
            factors: list[str]
    ) -> pd.Series:
        """计算当期 Rank IC"""
        rank_ic = {}

        for factor in factors:
            valid_df = df[[factor, "pctChg"]].dropna()
            if len(valid_df) < 2:
                corr = np.nan
            else:
                corr = spearmanr(
                    valid_df[factor], valid_df["pctChg"]
                ).correlation
            rank_ic[factor] = corr

        return pd.Series(rank_ic)

    @classmethod
    def _calc_rank_ics(
            cls,
            factors_data,
            factors_name
    ) -> pd.DataFrame:
        """计算Rank IC"""
        return pd.concat([
            cls.__calc_rank_ic(df, factors_name[date]).rename(date)
            for date, df in factors_data.items()
        ], axis=1).T

    # --------------------------
    # 权重方法
    # --------------------------
    @classmethod
    def _get_equal(
            cls,
            factors_name: dict[str, list[str]],
    ) -> pd.DataFrame:
        """
        等权权重
        :param factors_name: T期因子名
        """
        weights = pd.concat([
            pd.Series(
                1 / len(factors),
                index=factors,
                name=date
            ) for date, factors in factors_name.items()
        ], axis=1).T

        return weights

    @classmethod
    def _get_ic_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            window: int
    ) -> pd.DataFrame:
        """
        ic加权权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param window: 滚动窗口数
        :return 因子权重
        """
        # -1 计算 rank ic
        rank_ic = cls._calc_rank_ics(factors_data, factors_name)

        # -2 计算滚动值
        rolling_ic = rank_ic.rolling(window, min_periods=1).mean().abs()

        # -3 计算权重
        weights = rolling_ic.div(rolling_ic.sum(axis=1), axis="index")

        return weights

    @classmethod
    def _get_ir_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            window: int
    ) -> pd.DataFrame:
        """
        ir加权权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param window: 滚动窗口数
        :return 因子权重
        """
        # -1 计算 rank ic
        rank_ic = cls._calc_rank_ics(factors_data, factors_name)

        # -2 计算滚动 ir
        rolling_ir = rank_ic.rolling(window, min_periods=1).apply(cls._calc_ir)

        # -3 计算权重
        weights = rolling_ir.div(rolling_ir.sum(axis=1), axis="index")

        return weights

    @classmethod
    def _get_ir_decay_weight(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            window: int
    ) -> pd.DataFrame:
        """
        ir自适应衰减加权权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param window: 半衰期
        :return 因子权重
        """
        # -1 计算 rank ic
        rank_ic = cls._calc_rank_ics(factors_data, factors_name)

        # -2 计算自适应衰减ir
        ic_ewm_mean = rank_ic.ewm(halflife=window, min_periods=1).mean()
        ic_ewm_std = rank_ic.ewm(halflife=window, min_periods=1).std()
        rolling_ir = (ic_ewm_mean / ic_ewm_std).abs().replace(np.inf, 0).fillna(0)

        # -3 计算权重
        weights = rolling_ir.div(rolling_ir.sum(axis=1), axis="index")

        return weights

    # --------------------------
    # 公开 API
    # --------------------------
    @classmethod
    @validate_literal_params
    def get_weights(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            method: FACTOR_WEIGHT,
            window: int,
    ) -> pd.DataFrame:
        """
        获取因子权重
        :param factors_data: 因子数据
        :param factors_name: T期因子名
        :param method: 权重方法
        :param window: 因子滚动均值窗口数
        :return 因子权重
        """
        handlers = {
            "equal": lambda: cls._get_equal(factors_name),
            "ic_weight": lambda: cls._get_ic_weight(factors_data, factors_name, window),
            "ir_weight": lambda: cls._get_ir_weight(factors_data, factors_name, window),
            "ir_decay_weight": lambda: cls._get_ir_decay_weight(factors_data, factors_name, window)
        }

        return handlers[method]()


####################################################
class PositionWeight:
    """仓位权重"""

    # --------------------------
    # 权重方法
    # --------------------------
    @classmethod
    def _get_equal(
            cls,
            factors_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        等权权重
        :param factors_data: 因子数据
        """
        weights = pd.concat([
            pd.Series(
                1 / df.shape[0],
                index=df.index,
                name=date
            ) for date, df in factors_data.items()
        ], axis=1).T

        return weights

    @classmethod
    def __power_sorting_weights(
            cls,
            tilde_s,
            p,
            q
    ):
        """
        基于标准化得分向量 \tilde{s}_{t,(n)} 和超参数 p, q 计算权重。

        参数:
            tilde_s (list[float]): 标准化得分向量（来自standardize_ranks）。
            p (float): 正得分幂参数（控制多头权重集中度）。
            q (float): 负得分幂参数（控制空头权重集中度）。

        返回:
            list[float]: 权重向量，多头权重和为1，空头权重和为-1。
        """
        # 分离正、负得分（中性得分直接置0）
        negative_scores = [s for s in tilde_s if s < 0]
        positive_scores = [s for s in tilde_s if s > 0]

        # 计算分母（负得分和正得分的幂和）
        sum_neg = sum(abs(s) ** q for s in negative_scores)
        sum_pos = sum(s ** p for s in positive_scores)

        # 计算每个股票的权重
        weights = []
        for s in tilde_s:
            if s < 0:
                weight = 0.0 if sum_neg == 0 else - (abs(s) ** q) / sum_neg
            elif s == 0:
                weight = 0.0
            else:
                weight = 0.0 if sum_pos == 0 else (s ** p) / sum_pos

            weights.append(weight)

        return weights

    @classmethod
    def _get_power_sorting_weights(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factor_name: str,
            distribution: tuple[int, int]
    ):
        """
        幂排序权重
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param distribution: 超参数
                        -1 (0, 0) 等权
                        -2 (1, 1) 线性
                        -3 (3, 3) 两头
                        -4 (int, int) 其他
        :return: 幂排序仓位权重
        """
        # 映射空间 -1 (0, 1)为纯多头；-2 (-1, 1) 为多空组合
        feature_range = (0, 1)

        # -1 标准化特征排名向量
        factor_rank = pd.concat([
            DataProcessor.normalization(
                df[factor_name].rank(),
                feature_range=feature_range
            ).rename(date)
            for date, df in factors_data.items()
        ], axis=1).T
        """
        不大正常，结果每次都不一样，而且有的数据明显不对
        """
        print(factor_rank)
        print(factor_rank.shape)
        print(dd)
        for date, df in factors_data.items():
            score_rank = DataProcessor.normalization(df[factor_name].rank(), feature_range=(0, 1))
            print(score_rank)
            print(dd)


        # -2 计算幂排序权重
        weight = cls.__power_sorting_weights(score_rank, 1, 1)

    """
    映射到-1 1 ，是对冲有多有空
    映射到 0 1， 就是纯粹多头了
    然后根据 p q两个超参，调整权重
    既可以组内，也可全部
    """
    # --------------------------
    # 公开 API
    # --------------------------
    @classmethod
    @validate_literal_params
    def get_weights(
            cls,
            factors_data: dict[str, pd.DataFrame],
            factor_name: str,
            method: str,
    ) -> pd.DataFrame:
        """
        获取仓位权重
        :param factors_data: 因子数据
        :param factor_name: 排序因子名
        :param method: 权重方法
        :return 仓位权重
        """
        handlers = {
            "equal": lambda: cls._get_equal(factors_data),
        }

        cls._get_power_sorting_weights(factors_data, factor_name)

        print(dd)

        return handlers[method]()


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
    # 多因子线性回归
    # --------------------------
    @classmethod
    def calc_z_scores(
            cls,
            data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            weights: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """
        计算因子综合Z值
        :param data: 数据
        :param factors_name: T期因子名
        :param weights: 权重
        :return: 因子综合Z值
        """
        # -1 计算z值
        z_score = {
            date: filtered_df.rename("综合Z值")
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
            date: cls.processor.standardization(df, error="ignore")
            for date, df in z_score.items()
        }

        return z_score

    @classmethod
    def calc_predict_return(
            cls,
            data: dict[str, pd.DataFrame],
            window: int = 12
    ) -> dict[str, pd.DataFrame]:
        """
        滚动窗口回归预测收益率
        :param data: T期截面数据
        :param window: 滚动窗口长度
        :return: 预期收益率
        """
        # 按日期排序并转换为列表
        sorted_dates = sorted(data.keys())
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
                df = data[date][["综合Z值", "pctChg"]].copy()
                df["date"] = date  # 添加时间标记
                train_dfs.append(df)
            train_data = pd.concat(train_dfs).dropna()

            # 准备训练数据
            x_train = sm.add_constant(train_data["综合Z值"], has_constant='add')
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
            predict_df = data[predict_date][["综合Z值"]].copy()

            # 生成预测特征
            x_predict = sm.add_constant(predict_df["综合Z值"], has_constant='add')

            # 执行预测
            predicted = model.predict(x_predict)
            predicted.name = "predicted"

            # 存储结果
            result[predict_date] = pd.concat([data[predict_date], predicted], axis=1)

        return result

