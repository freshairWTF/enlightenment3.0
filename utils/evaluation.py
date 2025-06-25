"""评估因子/模型"""
from collections import deque
from functools import reduce
from scipy import stats
from datetime import datetime
from statsmodels import api as sm

import numpy as np
import pandas as pd

from constant.quant import ANNUALIZED_DAYS
from utils.processor import DataProcessor


####################################################
class Evaluation:
    """评估"""

    def __init__(self):
        self.returns = ReturnMetrics
        self.ic = ICMetrics
        self.ratio = RatioMetrics
        self.test = TestMetrics
        self.monitor = MonitorMetrics


####################################################
class ReturnMetrics:
    """收益率指标"""

    @classmethod
    def calc_duration(
            cls,
            returns: pd.DataFrame,
            cycle: str
    ) -> float:
        return returns.shape[0] / ANNUALIZED_DAYS.get(cycle)

    @classmethod
    def calc_group_returns_with_fixed(
            cls,
            grouped_data: dict[str, pd.DataFrame],
            cycle: str,
            max_label: str,
            min_label: str,
            mode: str,
            reverse: bool = False,
            trade_cost: float = 0.0,
    ) -> pd.DataFrame:
        """
        计算分组收益率（固定交易费率）
        :param grouped_data: 分组数据
        :param cycle: 周期
        :param max_label: 最大值标签
        :param min_label: 最小值标签
        :param mode: 计算模式
        :param reverse: 对冲方向
        :param trade_cost: 交易费用
        :return 分组收益率
        """
        # 手续费率
        trade_cost /= ANNUALIZED_DAYS.get(cycle)

        result = {}
        for date, df in grouped_data.items():
            grouped_df = df.groupby("group", observed=False)

            if mode == "equal":
                returns = grouped_df.apply(
                    lambda y: y["pctChg"].mean()
                )
            elif mode == "mv_weight":
                returns = grouped_df.apply(
                    lambda y: (
                            y["pctChg"] * (y["市值"] / y["市值"].sum())
                    ).sum()
                )
            elif mode == "position_weight":
                returns = grouped_df.apply(
                    lambda y: (
                            y["pctChg"] * y["position_weight"]
                    ).sum()
                )
            else:
                raise ValueError(f"不支持该收益率计算模式: {mode}")

            # 减去交易费用
            result[date] = returns - trade_cost

        result = pd.DataFrame.from_dict(result, orient="index").sort_index(axis=1).fillna(0)
        result = result[sorted(result.columns, key=lambda x: int(x))]

        # 对冲收益率
        result["hedge"] = (
            result[min_label] - result[max_label] if reverse
            else result[max_label] - result[min_label]
        )

        return result * 100

    @classmethod
    def calc_group_returns_with_turnover(
            cls,
            grouped_data: dict[str, pd.DataFrame],
            max_label: str,
            min_label: str,
            mode: str,
            reverse: bool = False,
            trade_cost: pd.Series = 0.0,
    ) -> pd.DataFrame:
        """
        计算分组收益率（换手率计算交易费率）
        :param grouped_data: 分组数据
        :param max_label: 最大值标签
        :param min_label: 最小值标签
        :param mode: 计算模式
        :param reverse: 对冲方向
        :param trade_cost: 交易费用
        :return 分组收益率
        """
        result = {}
        for date, df in grouped_data.items():
            grouped_df = df.groupby("group", observed=False)

            if mode == "equal":
                returns = grouped_df.apply(
                    lambda y: y["pctChg"].mean()
                )
            elif mode == "mv_weight":
                returns = grouped_df.apply(
                    lambda y: (
                            y["pctChg"] * (y["市值"] / y["市值"].sum())
                    ).sum()
                )
            elif mode == "position_weight":
                returns = grouped_df.apply(
                    lambda y: (
                            y["pctChg"] * y["position_weight"]
                    ).sum()
                )
            else:
                raise ValueError(f"不支持该收益率计算模式: {mode}")

            result[date] = returns - trade_cost.loc[datetime.strptime(date, "%Y-%m-%d").date()]

        result = pd.DataFrame.from_dict(result, orient="index").sort_index(axis=1).fillna(0)
        result = result[sorted(result.columns, key=lambda x: int(x))]

        # 对冲收益率
        result["hedge"] = (
            result[min_label] - result[max_label] if reverse
            else result[max_label] - result[min_label]
        )

        return result * 100

    @classmethod
    def t_value(
            cls,
            returns: pd.DataFrame
    ) -> pd.Series:
        """
        计算t值：分组收益率做t检验，是否显著！=0
        在实证资产定价领域的研究中，为了判断投资组合是否获取超额收益，需要对组合收益率序列进行显著性检验，即考察其平均收益是否显著不为零。
        标准的做法是运用经典的t检验。然而，由于投资组合收益率序列具有自相关性和异方差性，未经处理的标准误差、p值乃至t统计量很可能会失真，
        导致结论的可靠性大打折扣。
        滞后数 = 4(T/100)^a，其中T代表时间序列包含的观测次数。对于不同的核函数选择，a的取值各异：若采用Bartlett核，a取值为2/9；
        若采用quadratic spectral核来估计自相关与异方差调整后的标准误差，则a取值为4/25。
        :param returns: 收益率
        :return t值
        """
        def _process_series(series) -> pd.Series:
            series = series.dropna()
            if series.empty:
                return pd.Series([np.nan], index=["const"], dtype=np.float64)
            # 滞后期数计算（Bartlett核a=2/9）
            lag = max(1, int(4 * (len(series) / 100) ** (2 / 9)))

            model = sm.OLS(
                series,
                sm.add_constant(np.ones(len(series)))
            ).fit(
                cov_type="HAC",
                cov_kwds={
                    "maxlags": lag,
                    "kernel": "bartlett"
                }
            )
            return pd.Series(
                data=[model.tvalues[0]],
                index=["const"],
                dtype=np.float64
            )

        return returns.apply(_process_series).loc["const"]

    # --------------------------
    # 收益率
    # --------------------------
    @classmethod
    def win_to_loss_ratio(
            cls,
            returns: pd.DataFrame | pd.Series
    ) -> pd.Series:
        """
        盈亏比 = 总盈利 / 总亏损
        :param returns: 收益率
        :return 盈亏比
        """
        return (returns[returns > 0].mean() / returns[returns < 0].mean()).abs()

    @classmethod
    def winning_rate(
            cls,
            returns: pd.DataFrame | pd.Series,
            reverse: bool = False
    ) -> pd.Series:
        """
        胜率
        :param returns: 收益率
        :param reverse: 反转
        :return 胜率
        """
        return (
            returns.lt(0).sum() / returns.shape[0] if reverse
            else returns.gt(0).sum() / returns.shape[0]
        )

    @staticmethod
    def cum_return(
            returns: pd.DataFrame | pd.Series
    ) -> pd.DataFrame | pd.Series:
        """累计收益率"""
        return ((returns / 100 + 1).cumprod() - 1) * 100

    @classmethod
    def highest_rolling_return(
            cls,
            returns: pd.DataFrame | pd.Series,
            cycle: str,
            window: int
    ) -> pd.Series:
        """
        滚动累计收益率 最高值
        :param returns: 收益率
        :param cycle: 周期
        :param window: 窗口数
        :return 滚动累计收益率
        """
        window *= ANNUALIZED_DAYS.get(cycle)
        cum_return = (
            returns.rolling(window=window, min_periods=window)
            .apply(lambda x: cls.cum_return(x).iloc[-1])
        )
        return cum_return.max()

    @classmethod
    def lowest_rolling_return(
            cls,
            returns: pd.DataFrame | pd.Series,
            cycle: str,
            window: int
    ) -> pd.Series:
        """
        滚动累计收益率 最低值
        :param returns: 收益率
        :param cycle: 周期
        :param window: 窗口数
        :return 滚动累计收益率
        """
        window *= ANNUALIZED_DAYS.get(cycle)
        cum_return = (
            returns.rolling(window=window, min_periods=window)
            .apply(lambda x: cls.cum_return(x).iloc[-1])
        )
        return cum_return.min()

    @classmethod
    def scenario_return(
            cls,
            returns: pd.DataFrame,
            cycle: str,
            best: bool = True
    ) -> pd.Series:
        """
        1年期最优/最差预期 = 算数平均值 +- 1.5 * 标准差
        :param returns: 收益率
        :param cycle: 周期
        :param best: 滚动窗口数
        return: (df) 1年期最优/最差预期
        """
        annualized_num = ANNUALIZED_DAYS.get(cycle)
        annualized_mean = returns.mean() * annualized_num
        annualized_std = returns.std() * np.sqrt(annualized_num)
        return annualized_mean + 1.5 * annualized_std if best else annualized_mean - 1.5 * annualized_std

    @classmethod
    def annualized_return(
            cls,
            returns: pd.DataFrame | pd.Series,
            cycle: str
    ) -> pd.Series:
        """
        年化复合增长率 = (末 / 初)**(1 / 年数) - 1
        :param returns: 收益率
        :param cycle: 周期
        """
        annualized_num = ANNUALIZED_DAYS.get(cycle)

        def _process_series(series):
            return series.mean() * annualized_num

        if isinstance(returns, pd.DataFrame):
            return returns.apply(_process_series)
        elif isinstance(returns, pd.Series):
            return _process_series(returns)
        else:
            raise TypeError("仅支持 pandas DataFrame/Series 类型输入")

    @classmethod
    def cagr(
            cls,
            cum_returns: pd.DataFrame | pd.Series,
            cycle: str
    ) -> pd.Series:
        """
        年化复合增长率 = (末 / 初)**(1 / 年数) - 1
        :param cum_returns: 累计收益率
        :param cycle: 周期
        """
        annualized_num = ANNUALIZED_DAYS.get(cycle)

        def _process_series(series):
            with np.errstate(divide="ignore", invalid="ignore"):
                print(series)
                print((series[-1] / series[0]))
                print(1 / (len(series) / annualized_num))
                print((series[-1] / series[0]) ** (1 / (len(series) / annualized_num)))
                return (series[-1] / series[0]) ** (1 / (len(series) / annualized_num)) - 1

        if isinstance(cum_returns, pd.DataFrame):
            return cum_returns.apply(_process_series)
        elif isinstance(cum_returns, pd.Series):
            return _process_series(cum_returns)
        else:
            raise TypeError("仅支持 pandas DataFrame/Series 类型输入")

    @classmethod
    def maximum_drawdown(
            cls,
            cum_returns: pd.DataFrame
    ) -> pd.Series:
        """
        最大回撤
        :param cum_returns: 累计收益率
        """
        net_values = cum_returns + 100
        cumulative_max = net_values.cummax().clip(lower=100)
        return (1 - (net_values / cumulative_max)).max() * 100

    @classmethod
    def maximum_drawdown_period(
            cls,
            returns: pd.DataFrame
    ) -> pd.Series:
        """
        最大连续回撤周期
        :param returns: 收益率
        """
        def _max_negative_streak(s: pd.Series) -> int:
            # 生成连续负收益标记块
            is_negative: pd.Series = s < 0
            diff: pd.Series = is_negative != is_negative.shift()
            blocks = diff.cumsum()
            # 计算每个负块的长度
            negative_blocks = blocks.where(is_negative)
            block_sizes = negative_blocks.groupby(negative_blocks, observed=False).size()
            return block_sizes.max() if not block_sizes.empty else 0

        return returns.apply(_max_negative_streak)

    @classmethod
    def trade_cost_ratio(
            cls,
            returns: pd.DataFrame,
            trade_cost: pd.DataFrame,
            cycle: str,
    ) -> pd.Series:
        """
        交易费用比率 = 交易费用 / 年化收益率均值
        :param returns: 收益率
        :param cycle: 周期
        :param trade_cost: 交易费用
        :return: 交易费用比率
        """
        annualized_num = ANNUALIZED_DAYS.get(cycle)
        return (
                (trade_cost.mean() * annualized_num)
                / (returns.mean() * annualized_num)
        ) * 100

    @classmethod
    def check_j_shape_feature(
            cls,
            grouped_data: dict[str, pd.DataFrame],
            group_label: tuple[str, str],
            return_col: str = 'pctChg',
    ) -> float:
        """
        检验 倒J型特征
        :param grouped_data: k期分组截面数据
        :param group_label: 最高组与次高组的组名
        :param return_col: 收益率列名
        :return: 是否具备倒J型特征，即最高组收益率低于次高组
        """
        alpha = 0.05

        # -1 计算差值
        high_label, second_label = group_label
        group_df = pd.DataFrame({
            "最高组": [df[df['group'] == high_label][return_col].mean() for df in grouped_data.values()],
            "次高组": [df[df['group'] == second_label][return_col].mean() for df in grouped_data.values()]
        })
        diffs = group_df["最高组"] - group_df["次高组"]

        try:
            # -2 正态性检验（Shapiro-Wilk）
            _, shapiro_p = stats.shapiro(diffs)

            # -3 自动选择检验方法
            if shapiro_p > 0.05:
                # 满足正态分布
                _, p_value = stats.ttest_1samp(diffs, popmean=0, alternative='less')
            else:
                # 不满足正态分布
                _, p_value = stats.wilcoxon(diffs, alternative='less', mode='approx')

            return p_value
        except Exception as e:
            print(f"倒J型特征检验有误: {e}")
            return 999


####################################################
class ICMetrics:
    """IC指标"""

    @classmethod
    def calc_ic(
            cls,
            data: dict[str, pd.DataFrame],
            factor_col: str,
            pnl_col: str
    ) -> pd.DataFrame:
        """
        IC值（排序）
        :param data: 面板数据
        :param factor_col: 因子列名
        :param pnl_col: 收益率列名
        :return 秩相关系数
        """
        return pd.DataFrame.from_dict(
            {
                date: stats.spearmanr(df[pnl_col], df[factor_col], nan_policy="omit").statistic
                for date, df in data.items()
            },
            orient="index",
            columns=["ic"]
        )

    @classmethod
    def calc_icir(
            cls,
            ic: pd.DataFrame,
            cycle: str
    ) -> pd.Series:
        """
        ICIR
        :param ic: ic值
        :param cycle: 周期
        """
        annualized_num = ANNUALIZED_DAYS.get(cycle)
        return (
            (ic.mean() * annualized_num) / (ic.std() * np.sqrt(annualized_num))
        )

    @classmethod
    def calc_ic_significance(
            cls,
            ic: pd.DataFrame,
            sig_value: float = 0.03
    ) -> pd.Series:
        """
        ic显著性
        :param ic: ic值
        :param sig_value: 显著临界值
        """
        return ic.abs().gt(sig_value).sum() / ic.shape[0]

    @classmethod
    def calc_ic_decay(
            cls,
            data: dict[str, pd.DataFrame],
            factor_col: str,
    ) -> pd.DataFrame:
        """计算ic衰退"""
        decay = {}
        for i in range(1, 13, 1):
            try:
                shifted_data = DataProcessor().refactor.shift_factors_value_for_dict(data, i)
                ic = cls.calc_ic(
                    shifted_data,
                    factor_col,
                    "pctChg"
                )
                decay[str(i)] = ic.mean()
            except ValueError:
                continue
        return pd.DataFrame(decay).T

    @classmethod
    def get_half_life(
            cls,
            ic_mean: float,
            ic_decay: pd.Series
    ) -> float:
        """获取IC半衰期"""
        if ic_mean == 0:
            return float("nan")

        # 布尔掩码
        threshold = ic_mean / 2
        mask = ic_decay <= threshold if ic_mean > 0 else ic_decay >= threshold

        # 筛选索引
        half_life = ic_decay.index[mask]

        return float("nan") if half_life.empty else float(half_life[0])


####################################################
class RatioMetrics:
    """比率指标"""

    @classmethod
    def sharpe_ratio(
            cls,
            returns: pd.DataFrame,
            cycle: str,
            risk_free_rate: float = 3.0
    ) -> pd.Series:
        """
        夏普比率 = （投资组合收益 - 无风险利率） / 标准差
        :param returns: 收益率
        :param cycle: 周期
        :param risk_free_rate: 无风险利率
        :return 夏普比率
        """
        annualized_num = ANNUALIZED_DAYS.get(cycle)
        if annualized_num is None:
            raise ValueError(f"Invalid cycle: {cycle}")

        # 单期无风险收益率
        period_risk_free = risk_free_rate / annualized_num
        # 超额收益率
        excess_returns = returns - period_risk_free
        return (
            (excess_returns.mean() * annualized_num)
            / (excess_returns.std() * np.sqrt(annualized_num))
        )

    @classmethod
    def information_ratio(
            cls,
            returns: pd.DataFrame,
            cycle: str,
            benchmark: pd.Series
    ) -> pd.Series:
        """
        信息比率 = （投资组合收益 - 基准收益率） / 标准差
        :param returns: 收益率
        :param cycle: 周期
        :param benchmark: 基准指数收益率
        :return 信息比率
        """
        annualized_num = ANNUALIZED_DAYS.get(cycle)
        if annualized_num is None:
            raise ValueError(f"Invalid cycle: {cycle}")

        # 超额收益率
        excess_returns = returns.sub(benchmark, axis=0)

        return (
            (excess_returns.mean() * annualized_num)
            / (excess_returns.std() * np.sqrt(annualized_num))
        )

    @classmethod
    def sortino_ratio(
            cls,
            returns: pd.DataFrame,
            cycle: str,
            risk_free_rate: float = 3.0
    ) -> pd.Series:
        """
        索提诺比率 = （投资组合收益 - 无风险利率） / 下行标准差
        :param returns: 收益率
        :param cycle: 周期
        :param risk_free_rate: 无风险利率
        :return 索提诺比率
        """
        annualized_num = ANNUALIZED_DAYS.get(cycle)
        if annualized_num is None:
            raise ValueError(f"Invalid cycle: {cycle}")

        # 单期无风险收益率
        period_risk_free = risk_free_rate / annualized_num
        # 超额收益率
        excess_returns = returns - period_risk_free

        return (
            (excess_returns.mean() * annualized_num)
            / (excess_returns[excess_returns < 0].std() * np.sqrt(annualized_num))
        )

    @classmethod
    def sterling_ratio(
            cls,
            returns: pd.DataFrame,
            max_drawdown: pd.Series,
            cycle: str,
    ):
        """
        斯特林比率 = 投资组合收益 / （最大回撤 + 10%）
        :param returns: 收益率
        :param cycle: 周期
        :param max_drawdown: 最大回撤
        :return 斯特林比率
        """
        annualized_num = ANNUALIZED_DAYS.get(cycle)
        if annualized_num is None:
            raise ValueError(f"Invalid cycle: {cycle}")

        return (
            (returns.mean() * annualized_num)
            / (max_drawdown + 10)
        )

    @classmethod
    def kelly_ratio(
            cls,
            winning_rate: pd.Series,
            wtl_ratio: pd.Series
    ) -> pd.Series:
        """
        凯利比率 = P - (1-P) / W
        P：胜率；W：盈亏比
        :param winning_rate: 胜率
        :param wtl_ratio: 盈亏比
        :return 凯利比率
        """
        return (
            winning_rate - (1 / winning_rate) / wtl_ratio
        )


####################################################
class TestMetrics:
    """检验指标"""

    @classmethod
    def jonckheere_terpstra_test(
            cls,
            returns: pd.DataFrame
    ) -> pd.Series:
        """
        Jonckheere-Terpstra趋势检验（5% 显著性水平下，单边 p值 <= 0.05）
        :param returns: 收益率
        """
        # 按组别顺序提取数据（确保列名排序正确）
        groups = [returns[col].values for col in returns.columns]

        n_groups = len(groups)
        n = [len(g) for g in groups]
        u = 0

        # 计算统计量U（比较所有i<j组的观测值对）
        for i in range(n_groups - 1):
            for j in range(i + 1, n_groups):
                for x in groups[i]:
                    for y in groups[j]:
                        if x < y:
                            u += 1
                        elif x == y:
                            u += 0.5

        # 计算均值和方差
        sum_n = sum(n)
        mu = (sum_n ** 2 - sum([ni ** 2 for ni in n])) / 4
        var = (sum_n ** 2 * (2 * sum_n + 3) - sum([ni ** 2 * (2 * ni + 3) for ni in n])) / 72

        # 标准化统计量并计算p值
        z = (u - mu) / np.sqrt(var)
        if z > 0:
            p_value = 1 - stats.norm.cdf(z)
            trend = "Increasing"
        else:
            p_value = stats.norm.cdf(z)
            trend = "Decreasing"

        return pd.Series({
            "JT_统计量": z,
            "JT_p值": p_value,
            "JT_趋势": trend
        })

    @classmethod
    def rank_corr_test(
            cls,
            annualized_returns: pd.Series
    ) -> float:
        """
        计算秩相关系数
        :param annualized_returns: 年化复合增长率
        :return: 秩相关系数
        """
        valid_series = annualized_returns.dropna()
        if valid_series.empty:
            return 0.0

        x = valid_series.rank().values.astype(np.float64)
        y = np.arange(1, len(valid_series)+1, dtype=np.float64)
        return stats.spearmanr(
            x,
            y, # ignore[arg-type]
            nan_policy="omit"
        ).statistic

    @classmethod
    def calc_r_squared(
            cls,
            processed_data: dict[str, pd.DataFrame],
            factors_name: list[str],
            dependent_variable: str
    ) -> pd.DataFrame:
        r_squared = {}

        for date, df in processed_data.items():
            x = sm.add_constant(df[factors_name], has_constant="add")
            y = df[dependent_variable]
            model = sm.OLS(y, x).fit()
            r_squared[date] = model.rsquared

        return pd.DataFrame.from_dict(
            r_squared,
            orient="index",
            columns=["r_squared"]
        )

    @classmethod
    def calc_beta_feature(
            cls,
            processed_data: dict[str, pd.DataFrame],
            factors_name: list[str],
            dependent_variable: str
    ) -> pd.DataFrame:
        """
        回归计算beta
        :param processed_data: 处理后的数据
        :param factors_name: 因子名（自变量）
        :param dependent_variable: 因变量
        :return: {
            mean（因子对收益的平均影响方向（正/负）和幅度）: pd.Series
            std（系数波动性）: pd.Series
            t_stats（检验平均系数是否显著不为零，5%显著性水平下，T值需 >= 2）: pd.Series
        }
        """
        beta = {}

        # ----------------------------
        # 线性拟合
        # ----------------------------
        for date, df in processed_data.items():
            x = sm.add_constant(df[factors_name], has_constant="add")
            y = df[dependent_variable]
            model = sm.OLS(y, x).fit()

            beta[date] = model.params

        beta = pd.DataFrame.from_dict(beta).T

        # ----------------------------
        # 计算特征
        # ----------------------------
        mean_beta = beta.mean()
        std_beta = beta.std()
        se_beta = std_beta / (len(beta) ** 0.5)
        t_stats = mean_beta / se_beta

        return pd.DataFrame.from_dict(
            {
                "mean": mean_beta,
                "std": std_beta,
                "t_stats": t_stats
            }
        )


####################################################
class MonitorMetrics:
    """
    因子检测：
        1）因子间相关性
        2）因子拥挤度：
        多头/多空（多头组代表选股能力，多空比值组代表识别度，过高的识别度意味着因子超买）：
            1）多头上行、多空平：因子被市场青睐；
            2）多头平、多空上行：其他因子被市场青睐；
            3）多头上行、多空上行：因子被市场炒作；
            4）多头平、多空平：因子不被市场青睐。
        拥挤度指标：
            1）估值价差（多头组/多空比值）：
                1、多头组多用作参照，代表绝对值；
                2、多空比值组为主力，代表因子辨识度。
            2）做空比率价差（缺少做空交易数据）；
            3）配对相关性（长期多空相关系数/多空相关系数）：
                1、长期多空相关系数代表长期因子识别度；
                2、多空相关系数突然下降，往往意味着未来一段时间因子辨识度的上升。
            4）因子波动率比值（多头组/多空比值）；
            5）因子换手率比值（多头组/多空比值）；
            6）累计收益率（多头组/多空比值）；
            7）对冲贝塔比率（多头组/多空比值）；
        3）VIF
    """

    processor = DataProcessor()

    @classmethod
    def _get_grouped_feature(
            cls,
            df: pd.DataFrame,
            factor_name: str,
            target_col: str,
            agg_func: str = "mean",
            clean: bool = False
    ) -> pd.Series:
        """
        分类取均值
        :param df: 截面数据
        :param factor_name: 分组因子名
        :param target_col: 取均值因子名
        :param agg_func: 聚合方法
        :param clean: 是否清洗数据
        :return: 分组聚合特征
        """
        df = df[[factor_name, target_col]].copy()

        df["group"] = pd.qcut(
            df[factor_name],
            q=3,
            labels=["low", "medium", "high"]
        )

        if clean:
            df[target_col] = cls.processor.winsorizer.percentile(df[target_col])

        return df.groupby("group", observed=False)[target_col].agg(agg_func)

    @classmethod
    def calc_corr(
            cls,
            processed_data: dict[str, pd.DataFrame],
            factors_col: list[str],
            recent_period: int | None = None
    ) -> pd.DataFrame:
        """
        计算因子间相关系数矩阵
        :param processed_data: 预处理数据
        :param factors_col: 因子列名
        :param recent_period: 最近n期
        :return 相关系数矩阵
        """
        corr = {
            date: df[factors_col].corr(method="spearman").get(factors_col)
            for date, df in processed_data.items()
        }

        if recent_period:
            recent_corr = dict(deque(corr.items(), maxlen=recent_period))
            return (
                    reduce(lambda x, y: x.add(y), recent_corr.values()) / len(recent_corr)
            )
        # 计算均值
        else:
            return (
                    reduce(lambda x, y: x.add(y), corr.values()) / len(corr)
            )

    @classmethod
    def calc_valuation_spread(
            cls,
            processed_data: dict[str, pd.DataFrame],
            factor_col: str,
            value_factor: str,
            window: int
    ) -> pd.DataFrame:
        """
        计算估值价差：
            1）高估值价差：因子被过度交易、市场情绪过热，投资者追逐高因子得分股票。因子可能处于拥挤状态，未来收益可能下降。
            2）低估值价差：市场情绪平稳，投资者未过度追逐高因子得分股票。因子可能具有较高的潜在收益。
            3）估值价差衡量因子：1）市盈率；2）市净率；3）市销率；4）企业价值倍数；5）股息率
        注：选择与因子逻辑相关的估值指标。例如，价值因子通常使用市盈率或市净率。
        :param processed_data: 面板数据
        :param factor_col: 因子名
        :param value_factor: 估值因子名
        :param window: 窗口数
        :return 估值价差
        """
        valuation_spread = pd.DataFrame.from_dict(
            {
                date: cls._get_grouped_feature(
                    df,
                    factor_col,
                    value_factor,
                    agg_func="median",
                    clean=True
                )
                for date, df in processed_data.items()
            },
            orient="index"
        )
        valuation_spread = valuation_spread.rolling(window=window, min_periods=2).mean()
        valuation_spread["ratio"] = cls.processor.winsorizer.percentile(
            valuation_spread["high"] / valuation_spread["low"]
        )

        return valuation_spread

    @classmethod
    def calc_volatility(
            cls,
            processed_data: dict[str, pd.DataFrame],
            factor_col: str,
            window: int
    ) -> pd.DataFrame:
        """
        计算因子波动率：
            1）高因子波动率：因子被过度交易，导致多头端和空头端股票的表现趋同。因子可能处于拥挤状态，未来收益可能下降。
            2）低因子波动率：因子未被广泛关注，交易活跃度较低。因子可能具有较高的潜在收益。
            因子类型	低波动率范围	中等波动率范围	高波动率范围
            价值因子	< 0.10	0.10 - 0.20	> 0.20
            动量因子	< 0.15	0.15 - 0.25	> 0.25
            质量因子	< 0.12	0.12 - 0.22	> 0.22
            规模因子	< 0.18	0.18 - 0.30	> 0.30
        计算因子波动率比值：多空收益率标准差的比值
        :param processed_data: 面板数据
        :param factor_col: 因子名
        :param window: 窗口数
        :return 配对相关性
        """
        returns = pd.DataFrame.from_dict(
            {
                date: cls._get_grouped_feature(df, factor_col, "pctChg")
                for date, df in processed_data.items()
            },
            orient="index"
        )
        vol = returns.rolling(window=window, min_periods=2).std()
        vol["ratio"] = cls.processor.winsorizer.percentile(vol["high"] / vol["low"])

        return vol

    @classmethod
    def calc_turnover_ratio(
            cls,
            processed_data: dict[str, pd.DataFrame],
            factor_col: str,
            window: int,
    ) -> pd.DataFrame:
        """
        计算因子换手率比值：多空收益率换手率的比值
        :param processed_data: 预处理数据
        :param factor_col: 因子名
        :param window: 窗口数
        :return (pd.Series) 因子换手率
        """
        turn = pd.DataFrame.from_dict(
            {
                date: cls._get_grouped_feature(df, factor_col, "turn")
                for date, df in processed_data.items()
            },
            orient="index"
        )
        turn = turn.rolling(window=window, min_periods=2).mean()
        turn["ratio"] = cls.processor.winsorizer.percentile(turn["high"] / turn["low"])

        return turn

    @classmethod
    def calc_cum_return(
            cls,
            processed_data: dict[str, pd.DataFrame],
            factor_col: str,
            window: int
    ) -> pd.DataFrame:
        """
        计算因子累计收益率：
            1）高因子波动率：因子被过度交易，导致多头端和空头端股票的表现趋同。因子可能处于拥挤状态，未来收益可能下降。
            2）低因子波动率：因子未被广泛关注，交易活跃度较低。因子可能具有较高的潜在收益。
        :param processed_data: 预处理数据
        :param factor_col: 因子名
        :param window: 窗口数
        :return 累计收益率
        """
        returns = pd.DataFrame.from_dict(
            {
                date: cls._get_grouped_feature(df, factor_col, "pctChg")
                for date, df in processed_data.items()
            },
            orient="index"
        )
        cum = returns.rolling(window=window, min_periods=2).sum()
        cum["ratio"] = cls.processor.winsorizer.percentile(cum["high"] / cum["low"])

        return cum

    @classmethod
    def calc_pairwise_correlation(
            cls,
            shifted_data: dict[str, pd.DataFrame],
            factor_col: str,
            window: int
    ) -> pd.Series:
        """
        计算配对相关性：多空收益率的相关系数
        :param shifted_data: 平移数据
        :param factor_col: 因子名
        :param window: 窗口数
        :return 配对相关性
        """
        returns = pd.DataFrame.from_dict(
            {
                date: cls._get_grouped_feature(df, factor_col, "pctChg")
                for date, df in shifted_data.items()
            },
            orient="index"
        )
        return (
            returns["high"].rolling(window=window, min_periods=2)
            .corr(returns["low"])
        )

    # @classmethod
    # def calc_beta_ratio(
    #         cls,
    #         processed_data: dict[str, pd.DataFrame],
    #         factor_col: str,
    #         market_return: pd.DataFrame,
    #         window: int
    # ):
    #     """
    #     计算对冲贝塔比率：
    #         1）多空 Beta 比率接近 0：因子组合对市场风险的暴露较低，因子收益主要来自因子本身。
    #         2）多空 Beta 比率显著不为 0：因子组合可能受到市场整体波动的影响，需要进一步分析。
    #     :param processed_data: 预处理数据
    #     :param factor_col: 因子名
    #     :param window: 窗口数
    #     :return 对冲贝塔比率
    #     """
    #     def __calc_beta(df_):
    #         model = sm.OLS(df_.iloc[:, 0], df_.iloc[:, -1]).fit()
    #         beta_ = model.params.iloc[0]
    #         return beta_
    #
    #     returns = pd.DataFrame.from_dict(
    #         {
    #             date: cls._get_grouped_mean(df, factor_col, "pctChg")
    #             for date, df in processed_data.items()
    #         },
    #         orient="index"
    #     )
    #
    #     # 数据合并
    #     return_ = return_ * 100
    #     return_["market"] = market_return
    #     return_ = return_.astype(float)
    #
    #     # 计算对冲贝塔比率
    #
    #     return ratio
