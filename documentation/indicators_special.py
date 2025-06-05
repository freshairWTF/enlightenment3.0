"""
其他指标类
"""


import pandas as pd

from download import FUNCTION_OF_SPECIAL


###############################################################
class SpecialIndicators(object):
    """
    特殊指标：指标名（计算方法+变量） + 滚动窗口
    """
    def __init__(self, financial_df: pd.DataFrame, indicators_name: list, cycle: str):
        """
        :param cycle: 分析周期
        :param indicators_name: 指标名列表
        :param financial_df: 财务df
        """
        self.financial_df = financial_df
        self.indicators = pd.DataFrame(columns=financial_df.columns)
        self.indicators_name = indicators_name

        if cycle == 'year':
            self.window, self.min_period = 1, 1
        elif cycle == 'half':
            self.window, self.min_period = 2, 2
        # 月度数据的处理与季度实质上一致
        else:
            self.window, self.min_period = 4, 4

    def calculate(self):
        """
        计算财务指标
        """
        for ind_name in self.indicators_name:
            getattr(SpecialIndicators, FUNCTION_OF_SPECIAL[ind_name])(self)

        self.indicators = self.indicators.round(4)

    def _roic_rolling_mean_5y(self):
        """
        五年滚动平均资本回报率
        """
        window, min_period = self.window * 5, self.min_period * 5
        self.indicators.loc['资本回报率_5年滚动均值']\
            = self.financial_df.loc['资本回报率'].rolling(window=window, min_periods=min_period).mean()

    def _roic1_rolling_mean_5y(self):
        """
        五年滚动平均资本回报率
        """
        window, min_period = self.window * 5, self.min_period * 5
        self.indicators.loc['资本回报率1_5年滚动均值']\
            = self.financial_df.loc['资本回报率1'].rolling(window=window, min_periods=min_period).mean()
