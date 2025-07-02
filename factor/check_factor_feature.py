"""检验因子特征"""
from typing import Self
from scipy.stats import ttest_1samp
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import pandas as pd
import numpy as np

from constant.type_ import CYCLE
from utils.processor import DataProcessor


########################################################
class DifferentMarketAnalyzer:
    """不同市场分析"""

    def __init__(
            self,
            factor_ic: pd.Series,
            month_market_metrics: pd.DataFrame,
            day_market_metrics: pd.DataFrame,
            cycle: CYCLE
    ):
        """
        :param factor_ic: 因子ic
        :param month_market_metrics: 月频市场指标
        :param day_market_metrics: 日频市场指标
        :param cycle: 周期
        """
        self.cycle = cycle

        self.factor_ic = factor_ic
        self.factor_ic.index = pd.to_datetime(self.factor_ic.index)
        self.factor_ic = self.factor_ic.resample("M").mean()

        self.market_metrics = self.__gen_market_metrics(
            month_market_metrics,
            day_market_metrics
        )

    @classmethod
    def __gen_market_metrics(
            cls,
            month_market_metrics: pd.DataFrame,
            day_market_metrics: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        生成市场指标
        :param month_market_metrics: 月频市场指标
        :param day_market_metrics: 日频市场指标
        """
        month_market_metrics.index = month_market_metrics.index + pd.offsets.MonthEnd(0)
        return pd.concat(
            [
                month_market_metrics[["pctChg", "收盘价均线_0.25", "收盘价均线_1", "close"]],
                day_market_metrics.resample("M")['close'].apply(
                    lambda x: x.pct_change().std() * np.sqrt(12)
                ).rename("volatility")
            ],
            axis=1,
            join="inner"
        )

    @classmethod
    def __mark_status(
            cls,
            row: pd.Series,
            prev_state: str
    ) -> str:
        """
        判定市场状态
        :param row: 行数据
        :param prev_state: 之前状态
        """
        # 熊市
        if (
                (row['pctChg'] <= -0.06)
                or (row["close"] < row["收盘价均线_1"]
                    and row["收盘价均线_0.25"] < row["收盘价均线_1"]
                    and prev_state == 'Bear')
        ):
            return 'Bear'
        # 牛市
        elif (
                (row['pctChg'] >= 0.05
                 and row['volatility'] <= 0.12)
                or (row["close"] > row["收盘价均线_1"]
                    and row["收盘价均线_0.25"] > row["收盘价均线_1"]
                    and prev_state == 'Bull')
        ):
            return 'Bull'
        else:
            return 'Range'

    def __analyze_performance(
            self,
            status: pd.Series
    ) -> pd.DataFrame:
        """
        统计ic在各个市场环境下的表现
        :param status: 状态标识
        :return: 统计结果
        """
        # 数据合并
        merger_df = pd.concat(
            [self.factor_ic, status],
            axis=1,
            join="inner"
        )

        # 生成统计
        results = []
        for status, group in merger_df.groupby("status"):
            ic_mean = group["ic"].mean()
            volatility = group["ic"].std()

            # T检验
            t_stat, p_value = ttest_1samp(group["ic"], 0)

            results.append({
                'status': status,
                'mean': ic_mean,
                'vol': volatility,
                "ic_ir": ic_mean / volatility,
                't_stat': t_stat,
                'p_value': p_value,
                'n_obs': len(group)
            })

        return pd.DataFrame(results).set_index('status')

    # -----------------------------
    # 公开 API
    # -----------------------------
    def run(
            self
    ) -> pd.DataFrame:
        """运行"""
        # 初始赋值
        self.market_metrics["status"] = None
        # 首月前状态设为震荡市
        prev_state = 'Range'

        # 按时间顺序逐行处理（确保索引已按日期排序）
        for idx in self.market_metrics.index:
            row = self.market_metrics.loc[idx]
            current_state = self.__mark_status(row, prev_state)
            self.market_metrics.at[idx, 'status'] = current_state
            prev_state = current_state

        # 统计结果
        return self.__analyze_performance(self.market_metrics["status"])


########################################################
class MarkovChainAnalyzer:
    """
    基于马尔可夫链分析因子在不同市场状态（牛市/熊市/震荡市）中的表现差异
    1、 自动划分市场状态
    2、 分析因子在不同状态下的收益、波动性、显著性
    """

    def __init__(
            self,
            factor_return: pd.DataFrame,
            index_return: pd.DataFrame,
            cycle: CYCLE
    ):
        """
        :param factor_return: 因子收益率
        :param index_return: 指数收益率
        :param cycle: 周期
        """
        self.factor_return = factor_return
        self.index_return = index_return
        self.cycle = cycle

        # 市场状态数量 (默认为3: 牛市/熊市/震荡市)
        self.n_regimes = 3

        self.model = None
        self.state_prob = None
        self.performance = None

    def __preprocess_data(
            self
    ) -> Self:
        """数据预处理：处理缺失值、标准化"""
        # 日期转换：仅取年月
        if self.cycle in ["day", "week"]:
            self.factor_return.index = pd.to_datetime(self.factor_return.index)
            self.index_return.index = pd.to_datetime(self.index_return.index)
        else:
            self.factor_return.index = pd.to_datetime(self.factor_return.index).strftime("%Y-%m")
            self.index_return.index = pd.to_datetime(self.index_return.index).strftime("%Y-%m")

        # 去除缺失值
        self.factor_return.dropna(inplace=True)
        self.index_return.dropna(inplace=True)

        # 日期索引对齐
        self.factor_return, self.index_return = self.factor_return.align(
            self.index_return,
            join='inner',
            axis=0
        )
        print(self.factor_return)
        # 标准化
        print(self.index_return)
        self.index_return = DataProcessor().dimensionless.standardization(self.index_return)

        print(self.index_return)

        return self

    def __fit_model(
            self
    ) -> Self:
        """
        训练马尔可夫区制转换模型
        """
        self.model = MarkovRegression(
            endog=self.factor_return,
            exog=self.index_return,
            k_regimes=self.n_regimes,
            trend='c',
            switching_variance=True                     # 是否允许不同状态的方差不同
        )
        result = self.model.fit(disp=False)
        self.state_prob = result.smoothed_marginal_probabilities

        return self

    def __assign_states(
            self
    ) -> Self:
        """根据最大概率划分市场状态"""
        # 获取最可能的状态序列
        self.factor_return['state'] = np.argmax(self.state_prob, axis=1)

        # 自动生成状态标签 (假设按因子收益排序)
        state_means = self.factor_return.groupby('state')['factor_return'].mean()
        sorted_states = state_means.sort_values(ascending=False).index

        # 分配标签：牛市 > 震荡市 > 熊市
        label_map = {
            sorted_states[0]: 'Bull',
            sorted_states[1]: 'Neutral',
            sorted_states[2]: 'Bear'
        }
        self.factor_return['state_label'] = self.factor_return['state'].map(label_map)

        return self

    def __analyze_performance(
            self
    ) -> Self:
        """分析不同市场状态下的因子表现"""
        results = []
        for label, group in self.factor_return.groupby('state_label'):
            # 计算关键指标
            mean_return = group['factor_return'].mean()
            volatility = group['factor_return'].std()
            sharpe = mean_return / volatility if volatility != 0 else np.nan

            # T检验
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(group['factor_return'], 0)

            results.append({
                'state': label,
                'mean': mean_return,
                'vol': volatility,
                "ic_ir": mean_return / volatility,
                'sharpe_ratio': sharpe,
                't_stat': t_stat,
                'p_value': p_value,
                'n_obs': len(group)
            })

        self.performance = pd.DataFrame(results).set_index('state')

        return self

    # ------------------------
    # 公开 API
    # ------------------------
    def run_analysis(
            self
    ) -> Self:
        """执行完整分析流程"""
        return (
            self.__preprocess_data()
            .__fit_model()
            .__assign_states()
            .__analyze_performance()
        )
