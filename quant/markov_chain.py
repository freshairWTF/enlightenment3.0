from typing import Self
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constant.type_ import CYCLE


########################################################
class MarkovChainAnalyzer:
    """
    基于马尔可夫区制转换模型分析因子在不同市场状态（牛市/熊市/震荡市）中的表现差异

    1、 自动划分市场状态（使用马尔可夫区制转换模型）
    2、 分析因子在不同状态下的收益、波动性、显著性
    3、 可视化状态划分结果及因子表现
    """

    def __init__(
            self,
            factor_return: pd.Series,
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

        self.data = self.__preprocess_data()
        self.model = None
        self.state_prob = None
        self.performance = None

    def __preprocess_data(
            self
    ) -> pd.DataFrame:
        """数据预处理：处理缺失值、标准化"""
        # 日期转换：仅取年月
        if self.cycle in ["day", "week"]:
            self.factor_return.index = pd.to_datetime(self.factor_return.index)
            self.index_return.index = pd.to_datetime(self.index_return.index)
        else:
            self.factor_return.index = pd.to_datetime(self.factor_return.index).strftime("%Y-%m")
            self.index_return.index = pd.to_datetime(self.index_return.index).strftime("%Y-%m")

        # 数据合并
        merger_df = pd.concat(
            [
                self.factor_return.rename("factor_return"),
                self.index_return
            ],
            axis=1,
            join="inner"
        )

        # 去除缺失值
        merger_df.dropna(inplace=True)

        return merger_df

    def __fit_model(
            self
    ) -> Self:
        """
        训练马尔可夫区制转换模型
        """
        self.model = MarkovRegression(
            endog=self.data['factor_return'],
            exog=self.data[[ "累加收益率_0.25", "收益率标准差_0.25", "斜率_0.25"]],
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
        self.data['state'] = np.argmax(self.state_prob, axis=1)

        # 自动生成状态标签 (假设按因子收益排序)
        state_means = self.data.groupby('state')['factor_return'].mean()
        sorted_states = state_means.sort_values(ascending=False).index

        # 分配标签：牛市 > 震荡市 > 熊市
        label_map = {
            sorted_states[0]: 'Bull',
            sorted_states[1]: 'Neutral',
            sorted_states[2]: 'Bear'
        }
        self.data['state_label'] = self.data['state'].map(label_map)

        return self

    def __analyze_performance(
            self
    ) -> Self:
        """分析不同市场状态下的因子表现"""
        results = []
        for label, group in self.data.groupby('state_label'):
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

    def plot_results(
            self
    ) -> Self:
        """可视化状态划分及因子表现"""
        plt.figure(figsize=(14, 10))

        # 子图1：状态概率
        plt.subplot(2, 1, 1)
        for i in range(self.n_regimes):
            plt.plot(self.data.index, self.state_prob[i],
                     label=f'State {i} Prob', alpha=0.7)
        plt.title('Market Regime Probabilities')
        plt.legend()

        # 子图2：因子累计收益
        plt.subplot(2, 1, 2)
        self.data['cumulative_return'] = (
                (1 + self.data['factor_return']).cumprod() - 1
        )
        for state, color in zip(['Bull', 'Bear', 'Neutral'],
                                ['g', 'r', 'b']):
            subset = self.data[self.data['state_label'] == state]
            plt.scatter(subset.index, subset['cumulative_return'],
                        color=color, label=state, alpha=0.6)
        plt.title('Factor Cumulative Returns by Regime')
        plt.legend()

        plt.tight_layout()
        plt.show()
        return self

    # ------------------------
    # 公开 API
    # ------------------------
    def run_analysis(
            self
    ) -> Self:
        """执行完整分析流程"""
        return (
            self.__fit_model()
            .__assign_states()
            .__analyze_performance()
        )
