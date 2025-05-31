"""检验因子特征"""
from typing import Self
from scipy.stats import ttest_1samp
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.linear_model import RANSACRegressor, LinearRegression

import pandas as pd
import numpy as np
import statsmodels.api as sm

from constant.type_ import CYCLE
from utils.data_processor import DataProcessor


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
        self.index_return = DataProcessor.standardization(self.index_return)

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


########################################################
def check_u_shaped_feature(
        data_dict: dict[str, pd.DataFrame],
        factor_name: str,
        return_col: str = 'pctChg',  # 参数化收益率列名
        threshold: float = 0.7
) -> bool:
    from statsmodels.regression.quantile_regression import QuantReg

    valid_days = 0
    total_days = len(data_dict)

    condition_11 = 0
    condition_22 = 0
    condition_33 = 0
    condition_44 = 0

    for date, df in data_dict.items():
        try:
            # ===== 1. 数据预处理 =====
            df_clean = df.dropna(subset=[factor_name, return_col])
            if len(df_clean) < 30:
                continue

            x = df_clean[factor_name].values.reshape(-1, 1)
            y = df_clean[return_col].values

            # ===== 2. 二次项构造（关键优化）=====
            # 二次项用原始因子值计算（保持分布一致性）
            x_sq = x ** 2

            # 构建设计矩阵
            x_design = np.column_stack([x, x_sq])
            x_design = sm.add_constant(x_design)

            # ===== 3. 动态RANSAC拟合 =====
            # 基于MAD的动态残差阈值
            mad = np.median(np.abs(y - np.median(y)))
            residual_threshold = np.percentile(np.abs(y - np.median(y)), 75)

            model = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=max(0.5, 30 / len(x)),  # 提高样本要求
                residual_threshold=residual_threshold,
                max_trials=1000
            )
            model.fit(x_design, y)

            # ===== 4. 二次项显著性（验证存在弯曲） =====
            coefs = model.estimator_.coef_
            beta_1 = coefs[1]                   # 线性项系数
            beta_2 = coefs[2]                   # 二次项系数

            # 内点集OLS拟合获取p值
            inlier_mask = model.inlier_mask_
            ols_model = sm.OLS(y[inlier_mask], x_design[inlier_mask]).fit()
            p_value = ols_model.pvalues[2]
            condition_1 = (beta_2 != 0) and (p_value < 0.0005)

            # ===== 5. 端点斜率检验（验证两端趋势反向） =====
            extremum = -beta_1 / (2 * beta_2)

            # 极值点分位数位置（使用原始因子值）
            extremum_quantile = np.sum(x <= extremum) / len(x)
            left_ratio = np.sum(x <= extremum) / len(x)
            right_ratio = 1 - left_ratio

            # 条件2：极值点有效性（5%-95%分位且两侧均衡）
            q_low = np.percentile(x, 5)  # 5%分位值
            q_high = np.percentile(x, 95)  # 95%分位值
            is_central = (q_low <= extremum_quantile <= q_high)

            min_ratio = max(0.05, 8 / np.sqrt(len(x)))
            is_balanced = (min(left_ratio, right_ratio) > min_ratio)
            condition_2 = is_central and is_balanced

            # ===== 6. 拐点位置验证（验证转折点在数据内） =====
            model_low = QuantReg(y, sm.add_constant(x)).fit(q=0.3)
            model_high = QuantReg(y, sm.add_constant(x)).fit(q=0.7)
            slope_low = model_low.params[1]
            slope_high = model_high.params[1]
            condition_3 = (
                    ((slope_low < 0) and (slope_high > 0))  # U型
                    or
                    ((slope_low > 0) and (slope_high < 0))  # 倒U型
            )

            # ===== 7. 局部线性趋势检测（验证弯曲形态连贯） =====
            inlier_X = x[inlier_mask].flatten()
            inlier_y = y[inlier_mask]
            x_star = extremum

            # 左侧斜率（极值点左侧样本）
            left_mask = (inlier_X <= x_star)
            left_slope = np.polyfit(inlier_X[left_mask], inlier_y[left_mask], 1)[0] if sum(left_mask) > 10 else np.nan

            # 右侧斜率（极值点右侧样本）
            right_mask = (inlier_X > x_star)
            right_slope = np.polyfit(inlier_X[right_mask], inlier_y[right_mask], 1)[0] if sum(
                right_mask) > 10 else np.nan

            condition_4 = (
                    ((left_slope < 0) and (right_slope > 0))  # U型：左降右升
                    or
                    ((left_slope > 0) and (right_slope < 0))  # 倒U型：左升右降
            )

            # ===== 7. 核心条件判断 =====
            if all([condition_1, condition_2, condition_3, condition_4]):
                valid_days += 1

            if condition_1:
                condition_11 += 1
            if condition_2:
                condition_22 += 1
            if condition_3:
                condition_33 += 1
            if condition_4:
                condition_44 += 1

        except Exception as e:
            print(f"Error in {date}: {str(e)}")
            continue



    print(condition_11)
    print(condition_22)
    print(condition_33)
    print(condition_44)
    print(valid_days)
    print(total_days)
    pass_ratio = valid_days / total_days
    print(pass_ratio)
    print(pass_ratio >= threshold)

    return pass_ratio >= threshold