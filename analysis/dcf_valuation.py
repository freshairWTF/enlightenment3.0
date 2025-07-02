import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from matplotlib import pyplot as plt

"""
获取数据

计算 ebit预期增长率 贝叶斯模型 行业先验

给定 永续 简单
给定 wacc

计算折现值 
计算当前市值与折现值比值
"""

import pytensor
pytensor.config.exception_verbosity = 'high'

    # def _add_industry_constraints(self, pred_mu):
    #     # 添加未来预测约束作为潜在变量
    #     future_growth = pm.TruncatedNormal('future_growth',
    #                                        mu=pred_mu,
    #                                        sigma=1,
    #                                        lower=self._industry_stats['q20'],
    #                                        upper=self._industry_stats['q80'])
    #     return future_growth


# class BayesianDCFValuation:
#     """
#     贝叶斯自由现金流折现模型
#     """
#
#     def __init__(
#             self,
#             individual_data: pd.DataFrame,
#             industry_data: pd.DataFrame,
#             beta: float,
#             risk_free_rate: float = 0.03,
#             market_risk_premium: float = 0.045
#     ):
#         """
#         :param individual_data: 个股数据财务数据
#         :param industry_data: 行业财务数据
#         :param beta: 贝塔
#         :param risk_free_rate: 无风险利率
#         :param market_risk_premium: 市场风险补偿
#         """
#         self.individual_data = individual_data
#         self.industry_data = industry_data
#         self._validate_inputs()
#
#         self.beta = beta
#         self.risk_free_rate = risk_free_rate
#         self.market_risk_premium = market_risk_premium
#
#         self.model = None
#         self.trace = None
#         self.industry_priors = self._calculate_industry_priors()
#
#     # -----------------------------------
#     # 初始化方法
#     # -----------------------------------
#     def _validate_inputs(self):
#         """输入数据校验"""
#         required_individual = ["营业收入", "资产负债率", "税后利息率", "实际税率"]
#         required_industry = ["营业收入", "营收增速", "息税前利润率"]
#
#         missing_individual = [col for col in required_individual if col not in self.individual_data]
#         missing_industry = [col for col in required_industry if col not in self.industry_data]
#
#         if missing_individual:
#             raise ValueError(f"个股数据缺失字段: {missing_individual}")
#         if missing_industry:
#             raise ValueError(f"行业数据缺失字段: {missing_industry}")
#
#     def _calculate_industry_priors(self):
#         """计算行业先验参数"""
#         return {
#             indicator: {
#                 'mu': np.median(df),
#                 'sigma': np.median(np.abs(df - np.median(df))),
#                 'q20': np.percentile(df, 20),
#                 'q80': np.percentile(df, 80)
#             } for indicator, df in self.industry_data.items()
#         }
#
#     # -----------------------------------
#     # 建模、拟合、预测
#     # -----------------------------------
#     def _build_model(self):
#         """构建贝叶斯估值模型"""
#         with pm.Model() as self.model:
#
#             # ------------------------------
#             # 注册输入数据为模型变量
#             # ------------------------------
#             debt_ratio = pm.Data("debt_ratio", self.individual_data["资产负债率"].values)
#             tax_rate = pm.Data("tax_rate", self.individual_data["实际税率"].values)
#             after_tax_interest = pm.Data("after_tax_interest", self.individual_data["税后利息率"].values)
#             time = pm.Data('time', np.arange(len(self.individual_data)), mutable=True)
#
#             revenue_obs = pm.Data(
#                 "revenue_obs_data",
#                 self.individual_data['营业收入'].values
#             )
#             ebit_obs = pm.Data(
#                 "ebit_obs_data",
#                 self.individual_data['息税前利润'].values
#             )
#
#             n_periods = time.shape[0]
#
#             # ------------------------------
#             # EBIT 预测
#             # ------------------------------
#             # -1 ebit利润率
#             ebit_margin_mu = pm.Normal(
#                 'ebit_margin_mu',
#                 mu=self.industry_priors['息税前利润率']['mu'],
#                 sigma=self.industry_priors['息税前利润率']['sigma']
#             )
#             growth_rate = pm.Normal(
#                 'growth_rate',
#                 mu=0,
#                 sigma=0.1
#             )
#             ebit_margin = pm.Deterministic(
#                 'ebit_margin',
#                 ebit_margin_mu + growth_rate * time
#             )
#
#             # -2 营收增速
#             revenue_growth = pm.Normal(
#                 'revenue_growth',
#                 mu=self.industry_priors['营收增速']['mu'],
#                 sigma=self.industry_priors['营收增速']['sigma'],
#                 lower=-1.0,
#             )
#
#             # -3 预期收入
#             revenue = pm.GaussianRandomWalk(
#                 'revenue',
#                 mu=revenue_growth,
#                 sigma=0.01,
#                 init_dist=pm.Normal.dist(
#                     mu=self.individual_data['营业收入'].iloc[0],
#                     sigma=10
#                 ),
#                 shape=n_periods
#             )
#             pm.Normal(
#                 "revenue_obs_data",
#                 mu=revenue,
#                 sigma=0.05,
#                 observed=revenue_obs
#             )
#
#             # -4 预期ebit
#             ebit = pm.Deterministic(
#                 'ebit',
#                 revenue * ebit_margin
#             )
#             pm.Normal(
#                 "ebit_obs_data",
#                 mu=ebit,
#                 sigma=0.05,
#                 observed=ebit_obs
#             )
#
#             # ------------------------------
#             # WACC 计算
#             # ------------------------------
#             cost_of_equity = self.risk_free_rate + self.beta * self.market_risk_premium
#             wacc = pm.Deterministic(
#                 "wacc",
#                 (1 - debt_ratio) * cost_of_equity + debt_ratio * after_tax_interest * (1 - tax_rate)
#             )
#
#             # ------------------------------
#             # 现金流折现
#             # ------------------------------
#             # 自由现金流
#             fcf = pm.Deterministic(
#                 'fcf',
#                 ebit * (1 - tax_rate)
#             )
#
#             # 折现因子
#             discount_factors = pm.Deterministic(
#                 'discount_factors',
#                 1 / (1 + wacc) ** (time + 1)
#             )
#
#             # 折现值
#             pm.Deterministic(
#                 'present_value',
#                 pm.math.sum(fcf * discount_factors)
#             )
#
#             # 观测模型
#             pm.Normal(
#                 'ebit_obs',
#                 mu=ebit,
#                 sigma=0.1,
#                 observed=ebit_obs
#             )
#
#     def _fit(self, samples=4000, tune=2000, chains=4):
#         """执行MCMC采样"""
#         with self.model:
#             self.trace = pm.sample(
#                 draws=samples,
#                 tune=tune,
#                 chains=chains,
#                 target_accept=0.9,
#                 init='jitter+adapt_diag',
#                 return_inferencedata=True
#             )
#         return self.trace
#
#     def _forecast(self, forecast_years=5):
#         """生成未来预测"""
#         n_observed = len(self.individual_data)
#         # 扩展后的完整时间轴（0到n_observed + forecast_years -1）
#         extended_time = np.arange(n_observed + forecast_years)
#
#         # 扩展其他变量（假设未来值沿用末期观测值）
#         extended_debt_ratio = np.concatenate([
#             self.individual_data["资产负债率"].values,
#             np.full(forecast_years, self.individual_data["资产负债率"].iloc[-1])
#         ])
#         extended_tax_rate = np.concatenate([
#             self.individual_data["实际税率"].values,
#             np.full(forecast_years, self.individual_data["实际税率"].iloc[-1])
#         ])
#         extended_after_tax_interest = np.concatenate([
#             self.individual_data["税后利息率"].values,
#             np.full(forecast_years, self.individual_data["税后利息率"].iloc[-1])
#         ])
#         extended_ebit_obs = np.concatenate([
#             self.individual_data['息税前利润'].values,
#             np.full(forecast_years, np.nan)  # 预测期用NaN填充
#         ])
#
#         with self.model:
#             # 更新所有动态变量
#             pm.set_data({
#                 'time': extended_time,
#                 'debt_ratio': extended_debt_ratio,
#                 'tax_rate': extended_tax_rate,
#                 'after_tax_interest': extended_after_tax_interest,
#                 'ebit_obs_data': extended_ebit_obs  # 关键！更新观测数据维度
#             })
#
#             print("模型变量列表:")
#             for var in self.model.named_vars.values():
#                 print(f"{var.name}: shape={var.eval().shape}")
#
#             # # 检查输入数据是否合规
#             # self.model.check_test_point()
#
#
#             # 生成预测
#             forecast = pm.sample_posterior_predictive(
#                 self.trace,
#                 var_names=['fcf', 'wacc', 'revenue', 'present_value'],
#             )
#
#         return forecast
#
#
#     def _calculate_enterprise_value(self):
#         """计算企业价值"""
#         if self.trace is None:
#             raise ValueError("需要先执行fit()方法")
#
#         # 提取现值后验分布
#         pv_samples = self.trace.posterior['present_value'].values.flatten()
#         print( {
#             'mean': np.mean(pv_samples),
#             'median': np.median(pv_samples),
#             'hdi_95%': pm.hdi(pv_samples, 0.95),
#             'samples': pv_samples
#         })
#         # 计算估值指标
#         return {
#             'mean': np.mean(pv_samples),
#             'median': np.median(pv_samples),
#             'hdi_95%': pm.hdi(pv_samples, 0.95),
#             'samples': pv_samples
#         }
#
#     def analyze_results(self, show_plots=True, hdi_prob=0.94):
#         """分析模型结果并验证参数合理性
#
#         参数：
#         - show_plots: 是否显示诊断图表（默认True）
#         - hdi_prob: 最高密度区间概率（默认0.94）
#
#         返回：
#         - summary_df: 关键参数统计摘要DataFrame
#         """
#
#         # ======================
#         # 1. 基础结果查看
#         # ======================
#         print("=" * 40)
#         print("基础模型诊断")
#         print("=" * 40)
#
#         # 收敛性诊断
#         print(az.summary(self.trace, round_to=3))
#
#         # 绘制轨迹图
#         # if show_plots:
#         #     az.plot_trace(self.trace, compact=True, legend=True)
#         #     plt.suptitle("参数轨迹与分布图", y=1.05)
#         #     plt.tight_layout()
#         #     plt.show()
#
#         # ======================
#         # 2. 关键参数经济合理性分析
#         # ======================
#         print("\n" + "=" * 40)
#         print("关键参数经济合理性验证")
#         print("=" * 40)
#
#         # 定义合理范围（根据行业基准调整）
#         REASONABLE_RANGES = {
#             'ebit_margin_mu': (0.05, 0.20),  # 息税前利润率：5%-20%（假设制造业）
#             'growth_rate': (-0.05, 0.10),  # 增长率：-5%到+10%
#             'wacc': (0.08, 0.15),  # WACC：8%-15%
#             'revenue_growth': (0.03, 0.20),  # 营收增速：3%-20%（取对数前）
#         }
#
#         # 提取关键参数
#         params_to_check = ['ebit_margin_mu', 'growth_rate', 'wacc', 'revenue_growth']
#         summary_df = az.summary(self.trace, var_names=params_to_check, hdi_prob=hdi_prob)
#
#         # 验证每个参数
#         warnings = []
#         for param in params_to_check:
#             # 获取后验统计量
#             mean = summary_df.loc[param, 'mean']
#             hdi_low = summary_df.loc[param, f'hdi_{hdi_prob * 100 / 2}%']
#             hdi_high = summary_df.loc[param, f'hdi_{100 - hdi_prob * 100 / 2}%']
#             expected_range = REASONABLE_RANGES[param]
#
#             # 对数转换营收增速（因模型中对数处理）
#             # if param == 'revenue_growth':
#             #     mean = np.exp(mean) - 1e-6  # 逆转换
#             #     hdi_low = np.exp(hdi_low) - 1e-6
#             #     hdi_high = np.exp(hdi_high) - 1e-6
#             #     expected_range = (np.exp(expected_range[0]) - 1e-6,
#             #                       np.exp(expected_range[1]) - 1e-6)
#
#             # 检查是否在合理范围内
#             within_range = (hdi_low >= expected_range[0]) & (hdi_high <= expected_range[1])
#             status = "合理" if within_range else "警告"
#
#             # 输出结果
#             print(f"[{status}] {param}:")
#             print(f"  后验均值 = {mean:.3f}")
#             print(f"  {hdi_prob * 100:.0f}% HDI = [{hdi_low:.3f}, {hdi_high:.3f}]")
#             print(f"  预期范围 = {expected_range}\n")
#
#             if not within_range:
#                 warnings.append(f"参数 {param} 超出合理范围！")
#
#         # ======================
#         # 3. 现金流折现结果分析
#         # ======================
#         print("\n" + "=" * 40)
#         print("企业估值结果")
#         print("=" * 40)
#
#         # 提取现值后验分布
#         pv_samples = self.trace.posterior['present_value'].values.flatten()
#         pv_mean = np.mean(pv_samples)
#         pv_hdi = az.hdi(pv_samples, hdi_prob=hdi_prob)
#
#         print(f"企业现值（亿元）：")
#         print(f"  均值 = {pv_mean / 1e8:.2f} 亿")
#         print(f"  {hdi_prob * 100:.0f}% HDI = [{pv_hdi[0] / 1e8:.2f}, {pv_hdi[1] / 1e8:.2f}] 亿")
#
#         # 对比市场估值（假设用户提供）
#         if hasattr(self, 'market_cap'):
#             market_cap = self.market_cap
#             diff = (pv_mean - market_cap) / market_cap
#             print(f"\n与市场估值对比（假设市场值={market_cap / 1e8:.2f}亿）：")
#             print(f"  差异 = {diff * 100:.1f}%")
#             if abs(diff) > 0.3:
#                 warnings.append("模型估值与市场值差异超过30%！")
#
#         # ======================
#         # 4. 输出警告信息
#         # ======================
#         if warnings:
#             print("\n" + "!" * 50)
#             print("重要警告：")
#             for warn in warnings:
#                 print("  ⚠️ " + warn)
#             print("!" * 50)
#
#         return summary_df
#
#     # 示例调用
#     # model.analyze_results(show_plots=True)
#     # -----------------------------------
#     # 公开 API 方法
#     # -----------------------------------
#     def run(self) -> None:
#         """运行计算"""
#         # 初始化模型
#         self._build_model()
#
#         data_names = ["debt_ratio", "tax_rate", "after_tax_interest", "time", "ebit_obs_data"]
#         for name in data_names:
#             if name in self.model.named_vars:
#                 data = self.model[name].get_value()
#                 print(f"变量 {name} 的数据:\n", data)
#
#
#         # 参数估计
#         self._fit()
#         self.analyze_results(show_plots=True)
#
#         print(dd)
#
#         # 生成预测
#         self._forecast(forecast_years=5)
#         # 计算企业价值
#         valuation = self._calculate_enterprise_value()

class BayesianDCFValuation:
    """
    贝叶斯自由现金流折现模型
    """

    def __init__(
            self,
            individual_data: pd.DataFrame,
            industry_data: pd.DataFrame,
            beta: float,
            risk_free_rate: float = 0.03,
            market_risk_premium: float = 0.045
    ):
        """
        :param individual_data: 个股数据财务数据
        :param industry_data: 行业财务数据
        :param beta: 贝塔
        :param risk_free_rate: 无风险利率
        :param market_risk_premium: 市场风险补偿
        """
        self.individual_data = individual_data
        self.industry_data = industry_data

        self.beta = beta
        self.risk_free_rate = risk_free_rate
        self.market_risk_premium = market_risk_premium

        self.model = None
        self.trace = None
        self.industry_priors = self._calculate_industry_priors()

    # -----------------------------------
    # 初始化方法
    # -----------------------------------
    def _calculate_industry_priors(self):
        """计算行业先验参数"""
        return {
            indicator: {
                'mu': np.median(df),
                'sigma': np.median(np.abs(df - np.median(df))),
                'q20': np.percentile(df, 20),
                'q80': np.percentile(df, 80)
            } for indicator, df in self.industry_data.items()
        }

    # -----------------------------------
    # 建模、拟合、预测
    # -----------------------------------
    def _build_model(self):
        """构建贝叶斯估值模型"""
        with pm.Model() as self.model:

            # ------------------------------
            # 注册输入数据为模型变量
            # ------------------------------
            debt_ratio = pm.Data("debt_ratio", self.individual_data["资产负债率"].values)
            after_tax_interest = pm.Data("after_tax_interest", self.individual_data["税后利息率"].values)
            time = pm.Data('time', np.arange(len(self.individual_data)), mutable=True)

            fcff_obs_data = pm.Data(
                "fcff_obs_data",
                self.individual_data['企业自由现金流'].values
            )

            # ------------------------------
            # 企业自由现金流
            # ------------------------------
            fcff_mu = pm.Normal(
                'fcff_mu',
                mu=self.industry_priors['企业自由现金流']['mu'],
                sigma=self.industry_priors['企业自由现金流']['sigma']
            )
            growth_rate = pm.Normal(
                'growth_rate',
                mu=0,
                sigma=0.1
            )
            fcff = pm.Deterministic(
                'fcff',
                fcff_mu + growth_rate * time
            )

            # ------------------------------
            # WACC 计算
            # ------------------------------
            cost_of_equity = self.risk_free_rate + self.beta * self.market_risk_premium
            wacc = pm.Deterministic(
                "wacc",
                (1 - debt_ratio) * cost_of_equity + debt_ratio * after_tax_interest
            )

            # ------------------------------
            # 现金流折现
            # ------------------------------
            # 折现因子
            discount_factors = pm.Deterministic(
                'discount_factors',
                1 / (1 + wacc) ** (time + 1)
            )

            # 折现值
            pm.Deterministic(
                'present_value',
                pm.math.sum(fcff * discount_factors)
            )

            # 观测模型
            pm.Normal(
                'fcff_obs',
                mu=fcff,
                sigma=0.1,
                observed=fcff_obs_data
            )

    def _fit(self, samples=4000, tune=2000, chains=4):
        """执行MCMC采样"""
        with self.model:
            self.trace = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=0.9,
                init='jitter+adapt_diag',
                return_inferencedata=True
            )
        return self.trace

    def _forecast(self, forecast_years=5):
        """生成未来预测"""
        n_observed = len(self.individual_data)
        # 扩展后的完整时间轴（0到n_observed + forecast_years -1）
        extended_time = np.arange(n_observed + forecast_years)

        # 扩展其他变量（假设未来值沿用末期观测值）
        extended_debt_ratio = np.concatenate([
            self.individual_data["资产负债率"].values,
            np.full(forecast_years, self.individual_data["资产负债率"].iloc[-1])
        ])
        extended_tax_rate = np.concatenate([
            self.individual_data["实际税率"].values,
            np.full(forecast_years, self.individual_data["实际税率"].iloc[-1])
        ])
        extended_after_tax_interest = np.concatenate([
            self.individual_data["税后利息率"].values,
            np.full(forecast_years, self.individual_data["税后利息率"].iloc[-1])
        ])
        extended_ebit_obs = np.concatenate([
            self.individual_data['息税前利润'].values,
            np.full(forecast_years, np.nan)  # 预测期用NaN填充
        ])

        with self.model:
            # 更新所有动态变量
            pm.set_data({
                'time': extended_time,
                'debt_ratio': extended_debt_ratio,
                'tax_rate': extended_tax_rate,
                'after_tax_interest': extended_after_tax_interest,
                'ebit_obs_data': extended_ebit_obs  # 关键！更新观测数据维度
            })

            print("模型变量列表:")
            for var in self.model.named_vars.values():
                print(f"{var.name}: shape={var.eval().shape}")

            # # 检查输入数据是否合规
            # self.model.check_test_point()


            # 生成预测
            forecast = pm.sample_posterior_predictive(
                self.trace,
                var_names=['fcf', 'wacc', 'revenue', 'present_value'],
            )

        return forecast


    def _calculate_enterprise_value(self):
        """计算企业价值"""
        if self.trace is None:
            raise ValueError("需要先执行fit()方法")

        # 提取现值后验分布
        pv_samples = self.trace.posterior['present_value'].values.flatten()
        print( {
            'mean': np.mean(pv_samples),
            'median': np.median(pv_samples),
            'hdi_95%': pm.hdi(pv_samples, 0.95),
            'samples': pv_samples
        })
        # 计算估值指标
        return {
            'mean': np.mean(pv_samples),
            'median': np.median(pv_samples),
            'hdi_95%': pm.hdi(pv_samples, 0.95),
            'samples': pv_samples
        }

    def analyze_results(self, show_plots=True, hdi_prob=0.94):
        """分析模型结果并验证参数合理性

        参数：
        - show_plots: 是否显示诊断图表（默认True）
        - hdi_prob: 最高密度区间概率（默认0.94）

        返回：
        - summary_df: 关键参数统计摘要DataFrame
        """

        # ======================
        # 1. 基础结果查看
        # ======================
        print("=" * 40)
        print("基础模型诊断")
        print("=" * 40)

        # 收敛性诊断
        print(az.summary(self.trace, round_to=3))

        # 绘制轨迹图
        # if show_plots:
        #     az.plot_trace(self.trace, compact=True, legend=True)
        #     plt.suptitle("参数轨迹与分布图", y=1.05)
        #     plt.tight_layout()
        #     plt.show()

        # ======================
        # 2. 关键参数经济合理性分析
        # ======================
        print("\n" + "=" * 40)
        print("关键参数经济合理性验证")
        print("=" * 40)

        # 定义合理范围（根据行业基准调整）
        REASONABLE_RANGES = {
            'ebit_margin_mu': (0.05, 0.20),  # 息税前利润率：5%-20%（假设制造业）
            'growth_rate': (-0.05, 0.10),  # 增长率：-5%到+10%
            'wacc': (0.08, 0.15),  # WACC：8%-15%
            'revenue_growth': (0.03, 0.20),  # 营收增速：3%-20%（取对数前）
        }

        # 提取关键参数
        params_to_check = ['ebit_margin_mu', 'growth_rate', 'wacc', 'revenue_growth']
        summary_df = az.summary(self.trace, var_names=params_to_check, hdi_prob=hdi_prob)

        # 验证每个参数
        warnings = []
        for param in params_to_check:
            # 获取后验统计量
            mean = summary_df.loc[param, 'mean']
            hdi_low = summary_df.loc[param, f'hdi_{hdi_prob * 100 / 2}%']
            hdi_high = summary_df.loc[param, f'hdi_{100 - hdi_prob * 100 / 2}%']
            expected_range = REASONABLE_RANGES[param]

            # 对数转换营收增速（因模型中对数处理）
            # if param == 'revenue_growth':
            #     mean = np.exp(mean) - 1e-6  # 逆转换
            #     hdi_low = np.exp(hdi_low) - 1e-6
            #     hdi_high = np.exp(hdi_high) - 1e-6
            #     expected_range = (np.exp(expected_range[0]) - 1e-6,
            #                       np.exp(expected_range[1]) - 1e-6)

            # 检查是否在合理范围内
            within_range = (hdi_low >= expected_range[0]) & (hdi_high <= expected_range[1])
            status = "合理" if within_range else "警告"

            # 输出结果
            print(f"[{status}] {param}:")
            print(f"  后验均值 = {mean:.3f}")
            print(f"  {hdi_prob * 100:.0f}% HDI = [{hdi_low:.3f}, {hdi_high:.3f}]")
            print(f"  预期范围 = {expected_range}\n")

            if not within_range:
                warnings.append(f"参数 {param} 超出合理范围！")

        # ======================
        # 3. 现金流折现结果分析
        # ======================
        print("\n" + "=" * 40)
        print("企业估值结果")
        print("=" * 40)

        # 提取现值后验分布
        pv_samples = self.trace.posterior['present_value'].values.flatten()
        pv_mean = np.mean(pv_samples)
        pv_hdi = az.hdi(pv_samples, hdi_prob=hdi_prob)

        print(f"企业现值（亿元）：")
        print(f"  均值 = {pv_mean / 1e8:.2f} 亿")
        print(f"  {hdi_prob * 100:.0f}% HDI = [{pv_hdi[0] / 1e8:.2f}, {pv_hdi[1] / 1e8:.2f}] 亿")

        # 对比市场估值（假设用户提供）
        if hasattr(self, 'market_cap'):
            market_cap = self.market_cap
            diff = (pv_mean - market_cap) / market_cap
            print(f"\n与市场估值对比（假设市场值={market_cap / 1e8:.2f}亿）：")
            print(f"  差异 = {diff * 100:.1f}%")
            if abs(diff) > 0.3:
                warnings.append("模型估值与市场值差异超过30%！")

        # ======================
        # 4. 输出警告信息
        # ======================
        if warnings:
            print("\n" + "!" * 50)
            print("重要警告：")
            for warn in warnings:
                print("  ⚠️ " + warn)
            print("!" * 50)

        return summary_df

    # 示例调用
    # model.analyze_results(show_plots=True)
    # -----------------------------------
    # 公开 API 方法
    # -----------------------------------
    def run(self) -> None:
        """运行计算"""
        # 初始化模型
        self._build_model()

        data_names = ["debt_ratio", "tax_rate", "after_tax_interest", "time", "ebit_obs_data"]
        for name in data_names:
            if name in self.model.named_vars:
                data = self.model[name].get_value()
                print(f"变量 {name} 的数据:\n", data)


        # 参数估计
        self._fit()
        self.analyze_results(show_plots=True)

        print(dd)

        # 生成预测
        self._forecast(forecast_years=5)
        # 计算企业价值
        valuation = self._calculate_enterprise_value()
