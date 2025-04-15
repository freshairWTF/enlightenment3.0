import numpy as np
import pandas as pd

from base_metrics import Metrics, depends_on
from constant.type_ import CYCLE, KLINE_SHEET, validate_literal_params


###############################################################
class ValuationMetrics(Metrics):
    """
    通用估值分析类，通过 factor_mode 区分一般分析和量化分析
    """

    @validate_literal_params
    def __init__(
            self,
            financial_data: pd.DataFrame,
            kline_data: pd.DataFrame,
            bonus_data: pd.DataFrame,
            shares_data: pd.DataFrame,
            cycle: CYCLE,
            methods: list[str],
            function_map: dict[str, str],
            kline_adjust: KLINE_SHEET = "backward_adjusted"
    ):
        """
        :param financial_data: 财务数据
        :param kline_data: 行情数据
        :param bonus_data: 分红数据
        :param shares_data: 总股本数据
        :param cycle: 周期
        :param methods: 需要实现的方法
        :param function_map: 已定义的方法对应方法名
        :param kline_adjust: k线复权方法
        """
        self.financial_data = financial_data
        self.bonus_data = bonus_data
        self.shares_data = shares_data
        self.metrics = pd.DataFrame(index=financial_data.index)

        self.cycle = cycle
        self.methods = methods
        self.function_map = function_map
        self.kline_adjust = kline_adjust

        self.annual_window = self._setup_window(self.cycle)
        # 获取复权调整前的k线数据
        self.kline_data = self._get_kline_data(kline_data.copy(deep=True))

    # --------------------------
    # 初始化方法
    # --------------------------
    def _get_kline_data(
            self,
            kline_data: pd.DataFrame
    ) -> pd.DataFrame:
        """计算复权调整后的收盘价"""
        if self.kline_adjust == "backward_adjusted":
            kline_data["close"] /= kline_data["adjust_factor"]
        return kline_data

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def calculate(
            self
    ) -> None:
        """计算接口"""
        for method in self.methods:
            if method in self.function_map:
                method_name = self.function_map.get(method)
                if method_name and hasattr(self, method_name):
                    getattr(self, method_name)()
                else:
                    raise ValueError(f"未实现的方法: {method}")
            else:
                raise ValueError(f"未定义的指标: {method}")

    # --------------------------
    # 估值指标 私有方法
    # --------------------------
    def _earnings_per_share(self) -> None:
        """每股收益 = 归母净利润 / 平均股本"""
        self.metrics["每股收益"] = self._safe_divide(
                self.financial_data["净利润"],
                self._calc_rolling(self.shares_data["shares"], 2, 2)
        )

    def _core_earnings_per_share(self) -> None:
        """核心每股收益 = 核心利润 / 平均股本"""
        self.metrics["每股核心利润"] = self._safe_divide(
            self.financial_data["核心利润"],
            self._calc_rolling(self.shares_data["shares"], 2, 2)
        )

    def _net_assets_per_share(self) -> None:
        """每股净资产 = 所有者权益 / 总股本"""
        self.metrics["每股净资产"] = self._safe_divide(
                self.financial_data["所有者权益"],
                self.shares_data["shares"]
        )

    def _sales_per_share(self) -> None:
        """每股销售额 = 营业收入 / 平均股本"""
        self.metrics["每股销售额"] = self._safe_divide(
                self.financial_data["营业收入"],
                self._calc_rolling(self.shares_data["shares"], 2, 2)
        )

    def _dividend_per_share(self) -> None:
        """每股分红 = 分红总额 / 总股本"""
        self.metrics["每股分红"] = self._safe_divide(
            self.bonus_data["dividend"],
            self.shares_data["shares"]
        )

    def _market_value(self) -> None:
        """市值 = 收盘价 * 总股本"""
        self.metrics["市值"] = self.kline_data["close"] * self.shares_data["shares"]

    @depends_on("市值")
    def _log_market_value(self) -> None:
        """对数市值 = ln(市值)"""
        self.metrics["对数市值"] = np.log(self.metrics["市值"])

    @depends_on("每股收益")
    def _pe_ratio(self) -> None:
        """市盈率 = 股价 / 每股收益（滚动）"""
        self.metrics["市盈率"] = self._safe_divide(
            self.kline_data["close"],
            self.metrics["每股收益"]
        )

    @depends_on("每股核心利润")
    def _pe_ratio_by_core_profit(self) -> None:
        """核心利润市盈率 = 股价 / 核心每股收益（滚动）"""
        self.metrics["核心利润市盈率"] = self._safe_divide(
            self.kline_data["close"],
            self.metrics["每股核心利润"]
        )

    @depends_on("核心利润市盈率")
    def _reciprocal_of_pe_ratio_by_core_profit(self) -> None:
        """核心利润市盈率倒数 = 1 / 核心利润市盈率"""
        self.metrics["核心利润市盈率倒数"] = 1 / self.metrics["核心利润市盈率"]

    @depends_on("每股净资产")
    def _pe_ratio_adjusted_by_average_roe(self) -> None:
        """周期市盈率 = 股价 / (平均ROE * 每股净资产)"""
        rolling_roe = self.financial_data["权益净利率"].rolling(window=10000, min_periods=1).mean()

        self.metrics["周期市盈率"] = self._safe_divide(
            self.kline_data["close"],
            (rolling_roe * self.metrics["每股净资产"])
        )

    @depends_on("周期市盈率")
    def _reciprocal_of_pe_ratio_adjusted_by_average_roe(self) -> None:
        """周期市盈率倒数 = 1 / 周期市盈率"""
        self.metrics["周期市盈率倒数"] = 1 / self.metrics["周期市盈率"]

    @depends_on("每股收益")
    def _ep_ratio(self) -> None:
        """盈利市值比 = 每股收益 / 股价"""
        self.metrics["盈利市值比"] = self._safe_divide(
            self.metrics["每股收益"],
            self.kline_data["close"]
        )

    @depends_on("每股核心利润")
    def _ep_ratio_by_core_profit(self) -> None:
        """核心利润盈利市值比 = 核心每股收益 / 股价"""
        self.metrics["核心利润盈利市值比"] = self._safe_divide(
            self.metrics["每股核心利润"],
            self.kline_data["close"]
        )

    @depends_on("每股净资产")
    def _pb_ratio(self) -> None:
        """市净率 = 股价 / 每股净资产"""
        self.metrics["市净率"] = self._safe_divide(
            self.kline_data["close"],
            self.metrics["每股净资产"]
        )

    @depends_on("市净率")
    def _reciprocal_of_pb_ratio(self) -> None:
        """市净率倒数 = 1 / 市净率"""
        self.metrics["市净率倒数"] = 1 / self.metrics["市净率"]

    @depends_on("每股净资产")
    def _bm_ratio(self) -> None:
        """账面市值比 = 每股净资产 / 股价"""
        self.metrics["账面市值比"] = self._safe_divide(
            self.metrics["每股净资产"],
            self.kline_data["close"]
        )

    @depends_on("每股销售额")
    def _ps_ratio(self) -> None:
        """市销率 = 股价 / 滚动每股销售额"""
        self.metrics["市销率"] = self._safe_divide(
            self.kline_data["close"],
            self.metrics["每股销售额"]
        )

    @depends_on("市销率")
    def _reciprocal_of_ps_ratio(self) -> None:
        """市销率倒数 = 1 / 市销率"""
        self.metrics["市销率倒数"] = 1 / self.metrics["市销率"]

    @depends_on("每股分红")
    def _dividend_yield(self) -> None:
        """股息率 = 每股分红 / 股价 * 100"""
        self.metrics["股息率"] = self._safe_divide(
            self.metrics["每股分红"],
            self.kline_data["close"],
            percentage=True
        )

    @depends_on("市盈率")
    def _pe_to_growth(self) -> None:
        """市盈增长率 = 市盈率 / 净利润增长率"""
        growth_rate = self._calc_growth_rate(
            self.financial_data["净利润"],
            self.annual_window
        )
        self.metrics["市盈增长率"] = self.metrics["市盈率"] / growth_rate

    @depends_on("市盈增长率")
    def _reciprocal_of_pe_to_growth(self) -> None:
        """市盈增长率倒数 = 1 / 市盈增长率"""
        self.metrics["市盈增长率倒数"] = 1 / self.metrics["市盈增长率"]

    @depends_on("市净率")
    def _effective_yield(self) -> None:
        """实际收益率 = 权益净利率 / 市净率 * 100"""
        self.metrics["实际收益率"] = self._safe_divide(
            self.financial_data["权益净利率"],
            self.metrics["市净率"],
            percentage=True
        )

    @depends_on("市净率")
    def _effective_yield_by_core_profit(self) -> None:
        """核心利润实际收益率 = 核心利润率 / 市净率 * 100"""
        self.metrics["核心利润实际收益率"] = self._safe_divide(
            self.financial_data["核心利润净利率"],
            self.metrics["市净率"],
            percentage=True
        )

    @depends_on("市值")
    def _enterprise_value(self) -> None:
        """企业价值 = 息税前利润 / (市值 + 负债 - 超额现金 + 少数股东权益)"""
        market_value = (
                self.metrics["市值"] + self.financial_data["负债合计"]
                - self.financial_data["超额现金"] + self.financial_data["少数股东权益"]
        )
        self.metrics["企业价值"] = self._safe_divide(
            self.financial_data["息税前利润"],
            market_value
        )

    @depends_on("市值")
    def _pcf_ratio(self) -> None:
        """市现率 = 市值 / 经营现金流净额"""
        self.metrics["市现率"] = self._safe_divide(
            self.metrics["市值"],
            self.financial_data["经营活动产生的现金流量净额"]
        )

    @depends_on("市现率")
    def _reciprocal_of_pcf_ratio(self) -> None:
        """市现率倒数 = 1 / 市现率"""
        self.metrics["市现率倒数"] = 1 / self.metrics["市现率"]

    def _present_value_of_fcff(self) -> None:
        """基于滚动窗口的现值计算"""
        # ======================
        # 参数配置
        # ======================
        perpetual_growth = 0.03
        forecast_years = 5
        risk_free_rate = 0.03
        market_risk_premium = 0.09
        min_window = 5 * self.annual_window

        # ======================
        # 数据合成
        # ======================
        merger_df = pd.concat(
            [
                self.kline_data["市场贝塔_1"],
                self.financial_data[["企业自由现金流", "负债合计", "税后利息率", "营业收入"]],
                self.metrics["市值"]
            ],
            axis=1
        ).sort_index().dropna(how="any")

        # ======================
        # 定义滚动计算函数
        # ======================
        def _calculate_pv(window_df: pd.DataFrame) -> float:
            """单个窗口的现值计算"""
            # 校验窗口长度
            if len(window_df) < min_window:
                return np.nan

            # 提取当前基准年数据
            current_year_data = window_df.iloc[-1]
            fcff_series = window_df["企业自由现金流"]

            # 计算资本成本（WACC）
            equity_value = current_year_data["市值"]
            debt_value = current_year_data["负债合计"]
            total_capital = equity_value + debt_value
            if total_capital <= 0:
                return np.nan

            beta = current_year_data["市场贝塔_1"]
            cost_equity = risk_free_rate + beta * market_risk_premium
            cost_debt = current_year_data["税后利息率"]
            wacc = (equity_value / total_capital) * cost_equity \
                   + (debt_value / total_capital) * cost_debt

            if wacc <= perpetual_growth:
                return np.nan

            # 计算增长率
            if (fcff_series <= 0).any():
                revenue = window_df["营业收入"]
                growth_rate = self._calc_growth_rate(revenue, min_window, True)
            else:
                growth_rate = self._calc_growth_rate(fcff_series, min_window, True)
            growth_rate = np.clip(growth_rate, 0, 0.15)

            # 预测现金流
            base_fcff = fcff_series.iloc[-1]
            forecast = [base_fcff * (1 + growth_rate) ** (t + 1) for t in range(forecast_years)]
            discount = [1 / (1 + wacc) ** (t + 1) for t in range(forecast_years)]

            # 显性期现值 + 终值现值
            explicit_pv = sum(f * disc for f, disc in zip(forecast, discount))
            terminal_value = (forecast[-1] * (1 + perpetual_growth)) / (wacc - perpetual_growth)
            terminal_pv = terminal_value * discount[-1]

            return explicit_pv + terminal_pv

        # ======================
        # 执行滚动计算
        # ======================
        self.metrics["企业自由现金流折现值"] = pd.Series({
            df.index[-1]: _calculate_pv(df)
            for df in merger_df.rolling(window=min_window, min_periods=min_window)
        }).ffill()
