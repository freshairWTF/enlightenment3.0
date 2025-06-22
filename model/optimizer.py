"""优化器"""
from numpy import ndarray
from pandas import DataFrame
from pypfopt import EfficientFrontier, objective_functions, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
# from skfolio import (
#     Population, Portfolio, RiskMeasure,
#     PerfMeasure, RatioMeasure
# )
# from skfolio.optimization import MeanRisk
# from skfolio.preprocessing import prices_to_returns
# from skfolio.model_selection import WalkForward
# from skfolio.moments import GerberCovariance, ShrunkMu, DenoiseCovariance
# from skfolio.prior import EmpiricalPrior

import numpy as np
import pandas as pd

from constant.quant import ANNUALIZED_DAYS
from constant.type_ import CYCLE, validate_literal_params


###################################################
class PortfolioOptimizer:
    """
    基于PyPortfolioOpt的股票仓位权重优化器
    功能：计算最优股票权重分配、风险分析、离散化仓位分配
    """

    @validate_literal_params
    def __init__(
            self,
            asset_prices: pd.DataFrame,
            cycle: CYCLE,
            expected_return_method: str = "mean_historical_return",
            cov_method: str = "sample_cov",
            shrinkage_target: str = "constant_variance"
    ):
        """
        初始化优化器
        :param asset_prices: 资产历史价格数据（DataFrame，索引为日期，列为股票代码）
        :param expected_return_method: 预期收益计算方法
            -1 "mean_historical_return"
            -2 "ema_historical_return"
            -3 "capm_return"
        :param cov_method: 协方差计算方法
            -1 "sample_cov",
            -2 "ledoit_wolf",
            -3 "exp_cov"
        :param shrinkage_target: 收缩目标（适用于 Ledoit-Wolf 收缩估计，计算协方差矩阵）
            -1 constant_variance 恒定方差
            -2 single_factor 单因素模型，基于CAPM模型
            -3 constant_correlation 恒定相关性系数
        """
        self.asset_prices = asset_prices
        self.shrinkage_target = shrinkage_target
        self.frequency = ANNUALIZED_DAYS.get(cycle, 252)

        # 预期收益率
        self.expected_returns = self._calculate_expected_returns(expected_return_method)
        # 协方差矩阵
        self.cov_matrix = self._calculate_covariance(cov_method)
        self.ef = None                                                  # 有效前沿对象
        self.weights = None                                             # 优化权重

    def _calculate_expected_returns(
            self,
            method: str,
            compounding: bool = False,
            span: int = 500
    ) -> pd.Series:
        """
        计算预期收益率
        :param compounding: -1 True -> 几何平均数 -2 False -> 算术平均数
        :parma span: 指数加权时间窗口数
        """
        if method == "ema_historical_return":
            return expected_returns.ema_historical_return(
                self.asset_prices,
                compounding=compounding,
                span=span,
                frequency=self.frequency
            )
        elif method == "capm_return":
            return expected_returns.capm_return(
                self.asset_prices,
                compounding=compounding,
                frequency=self.frequency
            )
        return expected_returns.mean_historical_return(
            self.asset_prices,
            compounding=compounding,
            frequency=self.frequency
        )

    def _calculate_covariance(
            self,
            method: str
    ) -> DataFrame | ndarray:
        """计算协方差矩阵"""
        if method == "sample_cov":
            return risk_models.sample_cov(
                self.asset_prices,
                frequency=self.frequency
            )
        elif method == "exp_cov":
            return risk_models.exp_cov(
                self.asset_prices,
                frequency=self.frequency
            )
        return risk_models.CovarianceShrinkage(
            self.asset_prices,
            frequency=self.frequency
        ).ledoit_wolf(
            shrinkage_target=self.shrinkage_target
        )

    def optimize_weights(
            self,
            objective: str = "max_sharpe",
            weight_bounds: tuple = (0, 0.2),
            constraints: list | None = None,
            gamma: float = 0.1,
            sector_mapper: dict[str, str] | None = None,
            sector_upper: dict[str, float] | None = None,
            sector_lower: dict[str, float] | None = None,
            clean: bool = False,
            cutoff: float = 0.01,
            risk_free_rate: float = 0,
            risk_aversion: float = 1,
            market_neutral: bool = False,
            target_volatility: float = 0.15,
            target_return: float = 0.2
    ) -> pd.Series:
        """
        执行权重优化
        :param objective: 优化目标
                            -1 "max_sharpe" 最小波动率
                            -2 "min_volatility" 最大夏普比率
                            -3 "efficient_risk" 在给定的目标风险下，使收益最大
                            -4 "efficient_return" 在给定的目标收益下，使风险最小
                            -5 "max_quadratic_utility" 使给定的二次方效用最大化
        :param weight_bounds: 单只股票权重范围
                                -1 纯多头/纯空头 -> 取值范围（0, 1）
                                        PS：纯空头优化时，预期收益率 * -1
                                -2 多空 -> 取值范围（-1, 1）
        :param gamma: L2正则化系数（防止过拟合）
        :param constraints: 约束项 -> 例 lambda x : x[0] == 0.02
                            PS: 要么是一个线性平等约束，要么是凸不平等约束
        :param sector_mapper: 映射字典（用于不同资产组的权重总和添加限制）
        :param sector_upper: 权重上限（用于不同资产组的权重总和添加限制）
        :param sector_lower: 权重下限（用于不同资产组的权重总和添加限制）
        :param clean: 是否清理微小权重
        :param cutoff: 清理微小权重上限
        :param risk_free_rate: 无风险利率（max_sharpe参数）
        :param risk_aversion: 风险厌恶（max_quadratic_utility参数），取值范围 -> （0，1）
        :param market_neutral: 是否市场中立（权重和可为0）（max_quadratic_utility参数）
                                    PS: 权重边界需小于0
        :param target_volatility: 目标波动率（efficient_risk参数）
                                    PS: 若目标波动率设置不合理，则优化器会返回非预期权重！！！
        :param target_return: 目标收益率（efficient_return参数）
                                    PS:若目标收益率设置不合理，则优化器会返回非预期权重！！！
        :return: 优化后的权重字典 {股票代码: 权重}
        """
        # -1 初始化有效前沿
        self.ef = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
            solver="CVXOPT",
            weight_bounds=weight_bounds,
            verbose=False
        )

        # maxiters=1000, abstol=1e-6, reltol=1e-5, feastol=1e-6, refinement=3

        # -2 加入约束
        # if constraints:
        #     for constraint in constraints:
        #         self.ef.add_constraint(constraint)
        #
        # if sector_mapper and (sector_upper or sector_lower):
        #     self.ef.add_sector_constraints(
        #         sector_mapper=sector_mapper,
        #         sector_upper=sector_upper,
        #         sector_lower=sector_lower
        #     )

        # 添加正则化防止过拟合
        if gamma:
            self.ef.add_objective(objective_functions.L2_reg, gamma=gamma)

        # -3 执行优化
        if objective == "max_quadratic_utility":
            self.ef.max_quadratic_utility(risk_aversion=risk_aversion, market_neutral=market_neutral)
        elif objective == "min_volatility":
            self.ef.min_volatility()
        elif objective == "efficient_risk":
            self.ef.efficient_risk(target_volatility=target_volatility)
        elif objective == "efficient_return":
            self.ef.efficient_return(target_return=target_return)
        else:
            self.ef.max_sharpe(risk_free_rate=risk_free_rate)

        # -4 清理微小权重
        if clean:
            raw_weights = pd.Series(self.ef.clean_weights(cutoff=cutoff))
            self.weights = raw_weights / raw_weights.sum()
        else:
            self.weights = pd.Series(self.ef.weights, index=self.asset_prices.columns)

        return self.weights

    def portfolio_performance(
            self
    ) -> tuple:
        """
        获取投资组合绩效指标
        :return: (预期年化收益, 年化波动率, 夏普比率)
        """
        if not self.ef:
            raise ValueError("请先执行optimize_weights()进行优化")
        return self.ef.portfolio_performance(
            verbose=True
        )

    def discrete_allocation(
            self,
            total_portfolio_value: float = 1000000,
            latest_prices: pd.Series = None
    ) -> dict:
        """
        离散化仓位分配（生成实际可交易的股数）
            PS: 需要使用未复权价格，仅用于实盘交易
        :param total_portfolio_value: 投资组合总价值
        :param latest_prices: 各股票最新价格（Series）
        :return: 字典 {股票代码: 股数}
        """
        if latest_prices is None:
            latest_prices = self.asset_prices.iloc[-1]  # 默认使用最后一天价格

        weights = self.weights[self.weights != 0]
        latest_prices = latest_prices[weights.index.tolist()]

        da = DiscreteAllocation(
            weights.to_dict(),
            latest_prices,
            total_portfolio_value=total_portfolio_value
        )
        alloc, leftover = da.lp_portfolio(verbose=True)

        return {
            "allocation": alloc,
            "剩余资金": leftover,
            "分配比例": sum(alloc.values()) / total_portfolio_value
        }

    def risk_report(
            self
    ) -> pd.DataFrame:
        """生成风险贡献分析报告"""
        # 计算风险贡献
        weights = np.array(list(self.weights.to_dict().values()))
        portfolio_vol = self.ef.portfolio_performance()[1]

        # 边际风险贡献
        marginal_risk = self.cov_matrix.dot(weights) / portfolio_vol
        risk_contrib = np.multiply(weights, marginal_risk)

        # 构建报告
        report = pd.DataFrame({
            "股票代码": self.weights.index.tolist(),
            "权重": weights,
            "风险贡献": risk_contrib,
            "风险占比": risk_contrib / risk_contrib.sum()
        }).sort_values("风险占比", ascending=False)

        return report


###################################################
# class PortfolioOptimizerSK:
#     """
#     基于skfolio的股票仓位优化器
#     功能：计算最优权重分配、风险分析、离散化仓位分配
#     """
#
#     @validate_literal_params
#     def __init__(
#             self,
#             asset_prices: pd.DataFrame,
#             cycle: CYCLE,
#             expected_return_method: str = "historical",
#             cov_method: str = "sample_cov",
#     ):
#         """
#         初始化优化器
#         :param asset_prices: 资产历史价格数据（DataFrame，索引为日期，列为股票代码）
#         :param expected_return_method: 预期收益计算方法
#             -1 "historical"
#             -2 "ema"
#             -3 "capm"
#         :param cov_method: 协方差计算方法
#             -1 "sample_cov",
#             -2 "gerber",
#             -3 "denoise"
#         """
#         self.asset_prices = asset_prices
#         self.returns = prices_to_returns(asset_prices)  # 自动转换收益率
#
#         # 配置参数映射
#         self.frequency = 252 if cycle == "daily" else 52 if cycle == "weekly" else 12
#         self.mu_estimator = self._get_mu_estimator(expected_return_method)
#         self.cov_estimator = self._get_cov_estimator(cov_method)
#
#         self.model: MeanRisk | None = None
#         self.weights: pd.Series | None = None
#         self.performance: tuple | None = None
#
#     def _get_mu_estimator(self, method: str):
#         """预期收益估计器"""
#         estimators = {
#             "historical": ShrunkMu(),
#             "ema": ShrunkMu(ema_span=500),
#             "capm": ShrunkMu(capm=True)
#         }
#         return estimators.get(method, ShrunkMu())
#
#     def _get_cov_estimator(self, method: str):
#         """协方差估计器"""
#         estimators = {
#             "sample_cov": EmpiricalPrior(),
#             "gerber": GerberCovariance(),
#             "denoise": DenoiseCovariance()
#         }
#         return estimators.get(method, EmpiricalPrior())
#
#     def optimize_weights(
#             self,
#             objective: str = "max_sharpe",
#             weight_bounds: Tuple[float, float] = (0, 0.2),
#             gamma: float = 0.1,
#             sector_constraints: Optional[Dict] = None,
#             clean_weights: bool = False,
#             cutoff: float = 0.01,
#             risk_free_rate: float = 0.02,
#             target_volatility: float = 0.15,
#             target_return: float = 0.2
#     ) -> pd.Series:
#         """
#         执行权重优化
#         :param objective: 优化目标
#             - "max_sharpe": 最大夏普比率
#             - "min_volatility": 最小波动率
#             - "efficient_risk": 目标风险下最大收益
#             - "efficient_return": 目标收益下最小风险
#         :param weight_bounds: 单资产权重上下限
#         :param gamma: L2正则化系数
#         :param sector_constraints: 行业约束 {行业名: (下限, 上限)}
#         :param clean_weights: 是否清理微小权重
#         :param cutoff: 权重清理阈值
#         :param risk_free_rate: 无风险利率
#         """
#         # 初始化优化模型
#         self.model = MeanRisk(
#             objective_function=self._get_objective(objective),
#             risk_measure=RiskMeasure.VARIANCE,
#             prior_estimator=EmpiricalPrior(
#                 mu_estimator=self.mu_estimator,
#                 covariance_estimator=self.cov_estimator
#             ),
#             min_weights=weight_bounds[0],
#             max_weights=weight_bounds[1],
#             l2_coef=gamma,
#             risk_free_rate=risk_free_rate,
#         )
#
#         # 添加行业约束
#         if sector_constraints:
#             self._add_sector_constraints(sector_constraints)
#
#         # 训练模型
#         self.model.fit(self.returns)
#
#         # 获取权重
#         raw_weights = pd.Series(self.model.weights_, index=self.asset_prices.columns)
#
#         # 清理权重
#         if clean_weights:
#             raw_weights = raw_weights[raw_weights.abs() > cutoff]
#             self.weights = raw_weights / raw_weights.sum()
#         else:
#             self.weights = raw_weights
#
#         return self.weights
#
#     def _get_objective(self, objective: str) -> ObjectiveFunction:
#         """目标函数映射"""
#         objectives = {
#             "max_sharpe": ObjectiveFunction.MAXIMIZE_RATIO,
#             "min_volatility": ObjectiveFunction.MINIMIZE_RISK,
#             "efficient_risk": ObjectiveFunction.EFFICIENT_RISK,
#             "efficient_return": ObjectiveFunction.EFFICIENT_RETURN
#         }
#         return objectives[objective]
#
#     def _add_sector_constraints(self, constraints: Dict):
#         """添加行业权重约束"""
#         sector_groups = []
#         for sector, (lower, upper) in constraints.items():
#             sector_groups.append({
#                 "names": [col for col in self.returns.columns if col.startswith(sector)],
#                 "min": lower,
#                 "max": upper
#             })
#         self.model.add_group_constraints(sector_groups)
#
#     def portfolio_performance(self) -> Tuple:
#         """获取投资组合绩效指标"""
#         if not self.model:
#             raise ValueError("请先执行optimize_weights()进行优化")
#         portfolio = self.model.predict(self.returns)
#         return (
#             portfolio.annualized_mean_return,
#             portfolio.annualized_volatility,
#             portfolio.sharpe_ratio
#         )
#
#     def discrete_allocation(
#             self,
#             total_value: float = 100000,
#             latest_prices: pd.Series = None
#     ) -> Dict:
#         """
#         离散化仓位分配
#         :param total_value: 组合总价值
#         :param latest_prices: 各资产最新价格
#         """
#         if latest_prices is None:
#             latest_prices = self.asset_prices.iloc[-1]
#
#         # 计算各资产分配金额
#         alloc_amount = self.weights * total_value
#         alloc_shares = (alloc_amount / latest_prices).astype(int)
#
#         # 计算剩余资金
#         allocated = (alloc_shares * latest_prices).sum()
#         leftover = total_value - allocated
#
#         return {
#             "allocation": alloc_shares.to_dict(),
#             "remaining": leftover,
#             "allocation_ratio": allocated / total_value
#         }
#
#     def risk_report(self) -> pd.DataFrame:
#         """生成风险贡献报告"""
#         if not self.model:
#             raise ValueError("请先执行optimize_weights()进行优化")
#
#         portfolio = self.model.predict(self.returns)
#         return pd.DataFrame({
#             "Asset": self.weights.index,
#             "Weight": self.weights.values,
#             "RiskContribution": portfolio.risk_contribution,
#             "PercentContribution": portfolio.risk_contribution / portfolio.risk_contribution.sum()
#         }).sort_values("PercentContribution", ascending=False)