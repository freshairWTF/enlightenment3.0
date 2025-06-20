"""优化器"""
from numpy import ndarray
from pandas import DataFrame
from pypfopt import EfficientFrontier, objective_functions, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt import exceptions

import numpy as np
import pandas as pd


###################################################
class FactorSynthesisOptimizer:
    """因子权重优化"""

    def __init__(
            self,
            factor_returns: pd.DataFrame,
            cov_method: str = "sample_cov",
            window: int = 126
    ):
        """
        :param factor_returns: 因子收益率（格式：时间序列，每列为一个因子）
        :param cov_method: 协方差矩阵计算方法（"sample_cov", "ledoit_wolf", "exp_cov"等）
        :param window: 滚动窗口长度（仅对滚动协方差有效）
        """
        self.factor_returns = factor_returns
        self.cov_matrix = self._calculate_covariance(method=cov_method, window=window)
        self.optimal_weights = None
        self.optimizer = None  # 优化器对象

    def _calculate_covariance(
            self,
            method: str,
            window: int
    ) -> pd.DataFrame:
        """计算协方差矩阵（PyPortfolioOpt内置方法）"""
        if method == "rolling":
            # 需自行实现滚动协方差（PyPortfolioOpt原生不支持）
            return self.factor_returns.rolling(window).cov().dropna()
        else:
            # 使用PyPortfolioOpt的协方差方法
            return risk_models.risk_matrix(
                self.factor_returns,
                method=method,
                returns_data=True
            )

    # ---------------------------
    # 优化接口
    # ---------------------------
    def optimize_weights(
            self,
            objective: str = "max_sharpe",
            max_weight: float = 0.3,
            risk_aversion: float = 1.0,
            gamma: float = 0.1
    ) -> dict:
        """
        执行权重优化
        :param objective: 目标函数类型（"max_sharpe", "hrp", "min_vol", "quadratic_utility"）
        :param max_weight: 单个因子权重上限
        :param risk_aversion: 风险厌恶系数（用于均值-方差模型）
        :param gamma: L2正则化系数
        """
        # 选择优化器类型
        if objective == "hrp":
            self.optimizer = HRPOptimizer(returns=self.factor_returns)
        else:
            mu = expected_returns.mean_historical_return(self.factor_returns)
            self.optimizer = EfficientFrontier(mu, self.cov_matrix)

        # 添加权重约束
        self.optimizer.add_constraint(lambda w: w <= max_weight)

        # 正则化（防止过拟合）
        self.optimizer.add_objective(objective_functions.L2_reg, gamma=gamma)

        # 选择优化目标
        if objective == "max_sharpe":
            weights = self.optimizer.max_sharpe()
        elif objective == "min_vol":
            weights = self.optimizer.min_volatility()
        elif objective == "quadratic_utility":
            weights = self.optimizer.max_quadratic_utility(risk_aversion=risk_aversion)
        elif objective == "hrp":
            weights = self.optimizer.hrp_portfolio()
        else:
            raise ValueError("不支持的优化目标")

        # 清理权重
        self.optimal_weights = self.optimizer.clean_weights()
        return self.optimal_weights

    def analyze_risk(self) -> pd.DataFrame:
        """风险贡献分析（PyPortfolioOpt原生支持风险贡献计算）"""
        if not self.optimal_weights:
            raise exceptions.OptimizationError("需先执行optimize_weights()")

        # 计算风险贡献（仅HRP优化器直接支持）
        if isinstance(self.optimizer, HRPOptimizer):
            rc = self.optimizer.risk_contribution(self.optimal_weights)
        else:
            # 手动计算通用风险贡献
            portfolio_vol = self.optimizer.portfolio_performance()[1]
            marginal_risk = np.dot(self.cov_matrix, list(self.optimal_weights.values())) / portfolio_vol
            rc = np.multiply(list(self.optimal_weights.values()), marginal_risk)

        risk_report = pd.DataFrame({
            "Weight": self.optimal_weights.values(),
            "RiskContribution": rc,
            "RiskPercent": rc / np.sum(rc)
        }, index=self.factor_returns.columns)
        return risk_report.sort_values("RiskPercent", ascending=False)

    # ---------------------------
    # 其他功能
    # ---------------------------
    def get_performance(self) -> tuple:
        """获取组合性能指标（收益率、波动率、夏普比率）"""
        return self.optimizer.portfolio_performance(verbose=True)

    def discrete_allocation(self, total_value: float) -> dict:
        """离散化分配（需因子对应可交易标的）"""
        latest_prices = self.factor_returns.iloc[-1]  # 假设最后一行是最新价格
        da = DiscreteAllocation(
            self.optimal_weights,
            latest_prices,
            total_portfolio_value=total_value
        )
        alloc, leftover = da.lp_portfolio()
        return {"allocation": alloc, "剩余资金": leftover}


###################################################
class PortfolioOptimizer:
    """
    基于PyPortfolioOpt的股票仓位权重优化器
    功能：计算最优股票权重分配、风险分析、离散化仓位分配
    """

    def __init__(
            self,
            asset_prices: pd.DataFrame,
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
        self.cov_method = cov_method
        self.exp_return_method = expected_return_method
        self.shrinkage_target = shrinkage_target
        self.ef = None                                      # 有效前沿对象
        self.weights = None                                 # 优化权重

    def _calculate_expected_returns(
            self,
            compounding: bool = False,
            span: int = 500
    ) -> pd.Series:
        """
        计算预期收益率
        :param compounding: -1 True -> 几何平均数 -2 False -> 算术平均数
        :parma span: 指数加权时间窗口数
        """
        if self.exp_return_method == "ema_historical_return":
            return expected_returns.ema_historical_return(
                self.asset_prices,
                compounding=compounding,
                span=span
            )
        elif self.exp_return_method == "capm_return":
            return expected_returns.capm_return(
                self.asset_prices,
                compounding=compounding,
            )
        return expected_returns.mean_historical_return(
            self.asset_prices,
            compounding=compounding
        )

    def _calculate_covariance(
            self
    ) -> DataFrame | ndarray:
        """计算协方差矩阵"""
        if self.cov_method == "sample_cov":
            return risk_models.sample_cov(
                self.asset_prices
            )
        elif self.cov_method == "exp_cov":
            return risk_models.exp_cov(
                self.asset_prices
            )
        return risk_models.CovarianceShrinkage(
            self.asset_prices
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
            risk_aversion: float = 1,
            market_neutral: bool = False,
            target_volatility: float = 0.15,
            target_return: float = 0.2
    ) -> dict:
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
        :param risk_aversion: 风险厌恶（max_quadratic_utility参数），取值范围 -> （0，1）
        :param market_neutral: 是否市场中立（权重和可为0）（max_quadratic_utility参数）
                                    PS: 权重边界需小于0
        :param target_volatility: 目标波动率（efficient_risk参数）
                                    PS: 若目标波动率设置不合理，则优化器会返回非预期权重！！！
        :param target_return: 目标收益率（efficient_return参数）
                                    PS:若目标收益率设置不合理，则优化器会返回非预期权重！！！
        :return: 优化后的权重字典 {股票代码: 权重}
        """
        # -1 计算预期收益率、协方差矩阵
        mu = self._calculate_expected_returns()
        s = self._calculate_covariance()

        # -2 初始化有效前沿
        self.ef = EfficientFrontier(mu, s, weight_bounds=weight_bounds)

        # -3 加入约束
        for constraint in constraints:
            self.ef.add_constraint(constraint)
        if sector_mapper and (sector_upper or sector_lower):
            self.ef.add_sector_constraints(
                sector_mapper=sector_mapper,
                sector_upper=sector_upper,
                sector_lower=sector_lower
            )

        # 添加正则化防止过拟合
        if gamma:
            self.ef.add_objective(objective_functions.L2_reg, gamma=gamma)

        # 执行优化
        if objective == "max_quadratic_utility":
            self.ef.max_quadratic_utility(risk_aversion=risk_aversion, market_neutral=market_neutral)
        elif objective == "min_volatility":
            self.ef.min_volatility()
        elif objective == "efficient_risk":
            self.ef.efficient_risk(target_volatility=target_volatility)
        elif objective == "efficient_return":
            self.ef.efficient_return(target_return=target_return)
        else:
            self.ef.max_sharpe(risk_free_rate=0)

        # 清理微小权重
        if clean:
            return self.ef.clean_weights(cutoff=0.01, rounding=4)
        else:
            return self.ef.weights

    def portfolio_performance(
            self
    ) -> tuple:
        """
        获取投资组合绩效指标
        :return: (预期年化收益, 年化波动率, 夏普比率)
        """
        if not self.ef:
            raise ValueError("请先执行optimize_weights()进行优化")
        return self.ef.portfolio_performance(verbose=True)

    def discrete_allocation(
            self,
            total_portfolio_value: float = 100000,
            latest_prices: pd.Series = None
    ) -> dict:
        """
        离散化仓位分配（生成实际可交易的股数）
        :param total_portfolio_value: 投资组合总价值
        :param latest_prices: 各股票最新价格（Series）
        :return: 字典 {股票代码: 股数}
        """
        if latest_prices is None:
            latest_prices = self.asset_prices.iloc[-1]  # 默认使用最后一天价格

        da = DiscreteAllocation(
            self.weights,
            latest_prices,
            total_portfolio_value=total_portfolio_value
        )
        alloc, leftover = da.lp_portfolio()
        return {
            "allocation": alloc,
            "剩余资金": leftover,
            "分配比例": sum(alloc.values()) / total_portfolio_value
        }

    def risk_report(self) -> pd.DataFrame:
        """生成风险贡献分析报告"""
        if not self.ef or not self.weights:
            raise ValueError("请先执行优化")

        # 计算风险贡献
        cov_matrix = self._calculate_covariance()
        weights = np.array(list(self.weights.values()))
        portfolio_vol = self.ef.portfolio_performance()[1]

        # 边际风险贡献
        marginal_risk = cov_matrix.dot(weights) / portfolio_vol
        risk_contrib = np.multiply(weights, marginal_risk)

        # 构建报告
        report = pd.DataFrame({
            "股票代码": list(self.weights.keys()),
            "权重": weights,
            "风险贡献": risk_contrib,
            "风险占比": risk_contrib / risk_contrib.sum()
        }).sort_values("风险占比", ascending=False)

        return report
