import numpy as np
import pandas as pd

from base_metrics import Metrics, depends_on
from constant.type_ import FINANCIAL_CYCLE, validate_literal_params


###############################################################
class FinancialMetrics(Metrics):

    @validate_literal_params
    def __init__(
            self,
            financial_data: pd.DataFrame,
            bonus_data: pd.DataFrame,
            cycle: FINANCIAL_CYCLE,
            methods: list[str],
            function_map: dict[str, str],
            de_extreme_method
    ):
        """
        :param financial_data: 财务数据
        :param bonus_data：分红数据
        :param cycle：周期
        :param methods: 需要实现的方法
        :param function_map: 已定义的方法对应方法名
        :param de_extreme_method: 去极值方法
        """
        self.metrics = financial_data
        self.bonus_data = bonus_data
        self.cycle = cycle
        self.methods = methods
        self.function_map = function_map
        self.de_extreme_method = de_extreme_method

        self.annual_window = self._setup_window(self.cycle)

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
    # 财务指标
    # --------------------------
    @depends_on("现金转换周期")
    def _operating_monetary_fund(self) -> None:
        """
        经营性货币资金 = 现金转化周期 / 365 * 本期营收

        Notes
        -----
        version changed: 2025.04.21
        计算公式从「历史平均货币资金占收入比」调整为「现金转化周期/365*营收」
        """
        usual_monetary_fund = self.metrics["现金转换周期"] / 365 * self.metrics["营业收入"]

        # 最大限限制（货币资金）
        usual_monetary_fund = np.minimum(usual_monetary_fund, self.metrics["货币资金"])
        # 最小值限制（0）
        usual_monetary_fund = np.maximum(usual_monetary_fund, 0)

        self.metrics["经营性货币资金"] = usual_monetary_fund

    @depends_on("经营性货币资金")
    def _financial_monetary_fund(self) -> None:
        """
        金融性货币资金 = 货币资金 - 经营性货币资金
        """
        self.metrics["金融性货币资金"] = self.metrics["货币资金"] - self.metrics["经营性货币资金"]

    def _excess_cash(self) -> None:
        """
        超额现金 = 期末现金及现金等价物余额 + 流动资产 - 流动负债
        """
        self.metrics["超额现金"] = self.metrics["期末现金及现金等价物余额"] + self.metrics["流动资产合计"] - self.metrics["流动负债合计"]

    @depends_on("经营性货币资金")
    def _operating_current_assets(self) -> None:
        """
        经营性流动资产 = 应收票据及应收账款 + 应收款项融资 + 预付款项 + 存货 + 其他流动资产 + 其他应收款 - 应收利息 - 应收股利
            + 一年内到期的非流动资产 + 合同资产 + 流动资产其他项目 + 经营性货币资金
        """
        self.metrics["经营性流动资产"] = (
                self.metrics["应收票据及应收账款"]
                + self.metrics["应收款项融资"]
                + self.metrics["预付款项"]
                + self.metrics["存货"]
                + self.metrics["其他流动资产"]
                + self.metrics["其他应收款"]
                - self.metrics["应收利息"]
                - self.metrics["应收股利"]
                + self.metrics["一年内到期的非流动资产"]
                + self.metrics["合同资产"]
                + self.metrics["流动资产其他项目"]
                + self.metrics["经营性货币资金"]
        )

    @depends_on("金融性货币资金")
    def _financial_current_assets(self) -> None:
        """
        金融性流动资产 = 交易性金融资产 + 衍生金融资产 + 金融性货币资金 + 应收利息 + 应收股利
        """
        self.metrics["金融性流动资产"] = (
                self.metrics["交易性金融资产"]
                + self.metrics["衍生金融资产"]
                + self.metrics["金融性货币资金"]
                + self.metrics["应收利息"]
                + self.metrics["应收股利"]
        )

    def _display_operating_long_term_assets(self) -> None:
        """
        展示用经营性长期资产 = 长期应收款 + 投资性房地产 + 固定资产净额 + 在建工程 + 无形资产 + 开发支出 + 商誉 + 使用权资产
            + 长期待摊费用 + 递延所得税资产 + 其他非流动资产 + 生产性生物资产 + 油气资产 + 长期股权投资 + 发放贷款及垫款 + 持有待售资产
        生产性生物资产（农业）；油气资产（油气开采）
        """
        self.metrics["展示用经营性长期资产"] = (
                self.metrics["长期应收款"]
                + self.metrics["投资性房地产"]
                + self.metrics["固定资产净额"]
                + self.metrics["在建工程"]
                + self.metrics["无形资产"]
                + self.metrics["开发支出"]
                + self.metrics["商誉"]
                + self.metrics["使用权资产"]
                + self.metrics["长期待摊费用"]
                + self.metrics["递延所得税资产"]
                + self.metrics["其他非流动资产"]
                + self.metrics["生产性生物资产"]
                + self.metrics["油气资产"]
                + self.metrics["长期股权投资"]
                + self.metrics["发放贷款及垫款"]
                + self.metrics["持有待售资产"]
        )

    def _operating_long_term_assets(self) -> None:
        """
        经营性长期资产 = 长期应收款 + 固定资产净额 + 在建工程 + 无形资产 + 开发支出 + 使用权资产
            + 长期待摊费用 + 其他非流动资产 + 生产性生物资产 + 油气资产 + 长期股权投资
        生产性生物资产（农业）；油气资产（油气开采）

        Notes
        -----
        version changed: 2025.04.21
        计算公式剔除：
            1、商誉
            2、投资性房地产
            3、递延所得税资产
            4、持有待售资产
            5、发放贷款及垫款
        """
        self.metrics["经营性长期资产"] = (
                self.metrics["长期应收款"]
                + self.metrics["固定资产净额"]
                + self.metrics["在建工程"]
                + self.metrics["无形资产"]
                + self.metrics["开发支出"]
                + self.metrics["使用权资产"]
                + self.metrics["长期待摊费用"]
                + self.metrics["其他非流动资产"]
                + self.metrics["生产性生物资产"]
                + self.metrics["油气资产"]
                + self.metrics["长期股权投资"]
        )

    def _financial_long_term_assets(self) -> None:
        """
        金融性长期资产 = 其他债权投资 + 其他权益工具投资 + 其他非流动金融资产
        """
        self.metrics["金融性长期资产"] = self.metrics["其他债权投资"] + self.metrics["其他权益工具投资"] + self.metrics["其他非流动金融资产"]

    @depends_on("经营性流动资产", "经营性长期资产")
    def _operating_assets(self) -> None:
        """
        经营性资产 = 经营性流动资产 + 经营性长期资产
        """
        self.metrics["经营性资产"] = self.metrics["经营性流动资产"] + self.metrics["经营性长期资产"]

    @depends_on("金融性流动资产", "金融性长期资产")
    def _financial_assets(self) -> None:
        """
        金融性资产 = 金融性流动资产 + 金融性长期资产
        """
        self.metrics["金融性资产"] = self.metrics["金融性流动资产"] + self.metrics["金融性长期资产"]

    def _proportion_of_monetary_fund(self) -> None:
        """
        货币资金占比 = 货币资金 / 负债和所有者权益
        """
        self.metrics["货币资金占比"] = self._safe_divide(
            self.metrics["货币资金"],
            self.metrics["负债和所有者权益"]
        )

    def _proportion_of_current_assets(self) -> None:
        """
        流动资产占比 = 流动资产合计 / 负债和所有者权益
        """
        self.metrics["流动资产占比"] = self._safe_divide(
            self.metrics["流动资产合计"],
            self.metrics["负债和所有者权益"]
        )

    @depends_on("经营性资产")
    def _proportion_of_operating_assets(self) -> None:
        """
        经营性资产占比 = 经营性资产 / 负债和所有者权益
        """
        self.metrics["经营性资产占比"] = self._safe_divide(
            self.metrics["经营性资产"],
            self.metrics["负债和所有者权益"]
        )

    @depends_on("金融性资产")
    def _proportion_of_financial_assets(self) -> None:
        """
        金融性资产占比 = 金融性资产 / 负债和所有者权益
        """
        self.metrics["金融性资产占比"] = self._safe_divide(
            self.metrics["金融性资产"],
            self.metrics["负债和所有者权益"]
        )

    def _financial_liabilities(self) -> None:
        """
        金融性负债 = 短期借款 + 交易性金融负债 + 一年内到期的非流动负债 + 长期借款 + 应付债券 + 租赁负债
            + 长期应付款 + 向中央银行借款 + 衍生金融负债 + 应付利息 + 应付股利
        """
        self.metrics["金融性负债"] = (
                self.metrics["短期借款"]
                + self.metrics["交易性金融负债"]
                + self.metrics["一年内到期的非流动负债"]
                + self.metrics["长期借款"]
                + self.metrics["应付债券"]
                + self.metrics["租赁负债"]
                + self.metrics["长期应付款"]
                + self.metrics["向中央银行借款"]
                + self.metrics["衍生金融负债"]
                + self.metrics["应付利息"]
                + self.metrics["应付股利"]
        )

    @depends_on("金融性负债")
    def _net_interest_bearing_liabilities(self) -> None:
        """
        净有息负债 = 有息负债−现金及现金等价物 = 金融性负债 - 期末现金及现金等价物余额
        """
        self.metrics["净有息负债"] = self.metrics["金融性负债"] - self.metrics["期末现金及现金等价物余额"]

    def _operating_liabilities(self) -> None:
        """
        经营性负债 = 应付票据及应付账款 + 预收款项 + 应付职工薪酬 + 应交税费 + 其他应付款 - 应付利息 - 应付股利 + 其他流动负债
            + 长期应付职工薪酬 + 递延所得税负债 + 其他非流动负债 + 合同负债 + 预计负债 + 递延收益
        """
        self.metrics["经营性负债"] = (
                self.metrics["应付票据及应付账款"]
                + self.metrics["预收款项"]
                + self.metrics["应付职工薪酬"]
                + self.metrics["应交税费"]
                + self.metrics["其他应付款"]
                - self.metrics["应付利息"]
                - self.metrics["应付股利"]
                + self.metrics["其他流动负债"]
                + self.metrics["长期应付职工薪酬"]
                + self.metrics["递延所得税负债"]
                + self.metrics["其他非流动负债"]
                + self.metrics["合同负债"]
                + self.metrics["预计负债"]
                + self.metrics["递延收益"]
        )

    def _operating_current_liabilities(self) -> None:
        """
        经营性流动负债 = 应付票据及应付账款 + 预收款项 + 应付职工薪酬 + 应交税费 + 其他应付款 - 应付利息 - 应付股利 + 其他流动负债 + 合同负债
        """
        self.metrics["经营性流动负债"] = (
                self.metrics["应付票据及应付账款"]
                + self.metrics["预收款项"]
                + self.metrics["应付职工薪酬"]
                + self.metrics["应交税费"]
                + self.metrics["其他应付款"]
                - self.metrics["应付利息"]
                - self.metrics["应付股利"]
                + self.metrics["其他流动负债"]
                + self.metrics["合同负债"]
        )

    def _operating_long_term_liabilities(self) -> None:
        """
        经营性长期负债 = 长期应付职工薪酬 + 预计负债 + 递延所得税负债 + 其他非流动负债 + 递延收益
        """
        self.metrics["经营性长期负债"] = (
                self.metrics["长期应付职工薪酬"]
                + self.metrics["预计负债"]
                + self.metrics["递延所得税负债"]
                + self.metrics["其他非流动负债"]
                + self.metrics["递延收益"]
        )

    @depends_on("经营性负债")
    def _proportion_of_operating_liabilities(self) -> None:
        """
        经营性负债占比 = 经营性负债 / 负债和所有者权益
        """
        self.metrics["经营性负债占比"] = self._safe_divide(
            self.metrics["经营性负债"],
            self.metrics["负债和所有者权益"]
        )

    @depends_on("金融性负债")
    def _proportion_of_financial_liabilities(self) -> None:
        """
        金融性负债占比 = 金融性负债 / 负债和所有者权益
        """
        self.metrics["金融性负债占比"] = self._safe_divide(
            self.metrics["金融性负债"],
            self.metrics["负债和所有者权益"]
        )

    def _shareholder_investment(self) -> None:
        """
        股东入资 = 实收资本 + 资本公积 - 库存股
        """
        self.metrics["股东入资"] = (
                self.metrics["实收资本"]
                + self.metrics["资本公积"]
                - self.metrics["库存股"]
        )

    def _profit_accumulation(self) -> None:
        """
        利润留存 = 盈余公积 + 未分配利润 + 专项储备 + 一般风险准备
        """
        self.metrics["利润留存"] = (
                self.metrics["盈余公积"]
                + self.metrics["未分配利润"]
                + self.metrics["专项储备"]
                + self.metrics["一般风险准备"]
        )

    @depends_on("股东入资")
    def _proportion_of_shareholder_investment(self) -> None:
        """
        股东入资占比 = 股东入资 / 负债和所有者权益
        """
        self.metrics["股东入资占比"] = self._safe_divide(
            self.metrics["股东入资"],
            self.metrics["负债和所有者权益"]
        )

    @depends_on("利润留存")
    def _proportion_of_profit_accumulation(self) -> None:
        """
        利润留存占比 = 利润留存 / 负债和所有者权益
        """
        self.metrics["利润留存占比"] = self._safe_divide(
            self.metrics["利润留存"],
            self.metrics["负债和所有者权益"]
        )

    def _working_capital(self) -> None:
        """
        营运资本 = 流动资产 - 流动负债
        """
        self.metrics["营运资本"] = self.metrics["流动资产合计"] - self.metrics["流动负债合计"]

    @depends_on("经营性流动负债", "经营性流动资产")
    def _operating_working_capital(self) -> None:
        """
        经营性营运资本 = 经营性流动资产 - 经营性流动负债
        """
        self.metrics["经营性营运资本"] = self.metrics["经营性流动资产"] - self.metrics["经营性流动负债"]

    @depends_on("经营性长期资产", "经营性长期负债")
    def _net_operating_long_term_assets(self) -> None:
        """
        净经营性长期资产 = 经营性长期资产 - 经营性长期负债
        """
        self.metrics["净经营性长期资产"] = self.metrics["经营性长期资产"] - self.metrics["经营性长期负债"]

    @depends_on("净负债")
    def _total_net_operating_assets(self) -> None:
        """
        净经营资产 = 净负债 + 所有者权益
        """
        self.metrics["净经营资产"] = self.metrics["净负债"] + self.metrics["所有者权益"]

    @depends_on("金融性负债", "金融性资产")
    def _net_liabilities(self) -> None:
        """
        净负债 = 金融性负债 - 金融性资产
        """
        self.metrics["净负债"] = self.metrics["金融性负债"] - self.metrics["金融性资产"]

    def _gross_profit(self) -> None:
        """
        毛利 =  营业收入 - 营业成本
        """
        self.metrics["毛利"] = self.metrics["营业收入"] - self.metrics["营业成本"]

    def _gross_margin(self) -> None:
        """
        毛利率 = (1 - 营业成本 / 营业收入)
        """
        self.metrics["毛利率"] = 1 - self._safe_divide(
            self.metrics["营业成本"],
            self.metrics["营业收入"]
        )

    def _business_tax_burden_rate(self) -> None:
        """
        营业税金负担率 = 营业税金及附加 / 营业收入
        """
        self.metrics["营业税金负担率"] = self._safe_divide(
            self.metrics["营业税金及附加"],
            self.metrics["营业收入"]
        )

    @depends_on("实际税率")
    def _core_profit(self) -> None:
        """
        核心利润 = (营业收入 - 营业成本 - 营业税金及附加 - 销售费用 - 管理费用  - 利息费用 - 研发费用) * (1 - 实际税率)
        PS：财务费用应为利息费用，但新浪财经无此科目
        """
        self.metrics["核心利润"] = (
                (
                        self.metrics["营业收入"]
                        - self.metrics["营业成本"]
                        - self.metrics["营业税金及附加"]
                        - self.metrics["销售费用"]
                        - self.metrics["管理费用"]
                        - self.metrics["利息费用"]
                        - self.metrics["研发费用"]
                )
                * (1 - self.metrics["实际税率"])
        )

    @depends_on("核心利润")
    def _core_profit_margin(self) -> None:
        """
        核心利润率 = 核心利润 / 营业收入
        """
        self.metrics["核心利润率"] = self._safe_divide(
            self.metrics["核心利润"],
            self.metrics["营业收入"]
        )

    @depends_on("实际税率")
    def _investment_profit(self) -> None:
        """
        投资利润 = (公允价值变动收益 + 投资收益) * (1 - 实际税率)
        """
        self.metrics["投资利润"] = (
                (self.metrics["公允价值变动收益"] + self.metrics["投资收益"])
                * (1 - self.metrics["实际税率"])
        )

    @depends_on("实际税率")
    def _miscellaneous_profit(self) -> None:
        """
        杂项利润 = (其他收益 + 资产处置收益) * (1 - 实际税率)
        """
        self.metrics["杂项利润"] = (
                (self.metrics["其他收益"] + self.metrics["资产处置收益"])
                * (1 - self.metrics["实际税率"])
        )

    @depends_on("实际税率")
    def _net_non_operating_income_and_expenditure(self) -> None:
        """
        营业外收支净额 = (营业外收入 - 营业外支出) * (1 - 实际税率)
        """
        self.metrics["营业外收支净额"] = (
                (self.metrics["营业外收入"] - self.metrics["营业外支出"])
                * (1 - self.metrics["实际税率"])
        )

    def _sales_profit_margin(self) -> None:
        """
        销售利润率 = 利润总额 / 营业收入
        """
        self.metrics["销售利润率"] = self._safe_divide(
            self.metrics["利润总额"],
            self.metrics["营业收入"]
        )

    @depends_on("实际税率")
    def _net_operating_profit(self) -> None:
        """
        经营净利润 = (营业收入 - 营业成本 - 营业税金及附加 - 管理费用 - 销售费用 - 研发费用 + 资产减值损失 + 信用减值损失
            + 其他收益 + 营业外收入 - 营业外支出 + 对联营企业和合营企业的投资收益) * (1 - 所得税税率)
        注：减值损失、公允价值变动收益分为经营与金融，现根据财报一刀切，将减值损失列为经营项目，公允价值变动收益列为金融项目
            投资收益中的对联营企业和合营企业的投资收益属于经营项目，其余为金融项目
        """
        self.metrics["经营净利润"] = (
                (
                        self.metrics["营业收入"]
                        - self.metrics["营业成本"]
                        - self.metrics["营业税金及附加"]
                        - self.metrics["管理费用"]
                        - self.metrics["销售费用"]
                        - self.metrics["研发费用"]
                        + self.metrics["资产减值损失"]
                        + self.metrics["信用减值损失"]
                        + self.metrics["其他收益"]
                        + self.metrics["营业外收入"]
                        - self.metrics["营业外支出"]
                        + self.metrics["对联营企业和合营企业的投资收益"]
                )
                * (1 - self.metrics["实际税率"])
        )

    @depends_on("实际税率")
    def _financial_net_profit(self) -> None:
        """
        金融净利润 = (投资收益 - 对联营企业和合营企业的投资收益 + 公允价值变动收益 - 财务费用) * (1 - 所得税税率)
        注：减值损失、公允价值变动收益分为经营与金融，现根据财报一刀切，将减值损失列为经营项目，公允价值变动收益列为金融项目
            投资收益中的对联营企业和合营企业的投资收益属于经营项目，其余为金融项目
        """
        self.metrics["金融净利润"] = (
                (
                        self.metrics["投资收益"]
                        - self.metrics["对联营企业和合营企业的投资收益"]
                        + self.metrics["公允价值变动收益"]
                        - self.metrics["财务费用"]
                )
                * (1 - self.metrics["实际税率"])
        )

    @depends_on("核心利润")
    def _proportion_of_core_profit(self) -> None:
        """
        核心利润占比 = 核心利润 / 利润总额
        """
        self.metrics["核心利润占比"] = self._safe_divide(
            self.metrics["核心利润"],
            self.metrics["利润总额"].abs()
        )

    def _sales_expense_rate(self) -> None:
        """
        销售费用率 = 销售费用 / 营业收入
        """
        self.metrics["销售费用率"] = self._safe_divide(
            self.metrics["销售费用"],
            self.metrics["营业收入"]
        )

    def _management_expense_rate(self) -> None:
        """
        管理费用率 = 管理费用 / 营业收入
        """
        self.metrics["管理费用率"] = self._safe_divide(
            self.metrics["管理费用"],
            self.metrics["营业收入"]
        )

    def _financial_expense_rate(self) -> None:
        """
        财务费用率 = 财务费用 / 营业收入
        """
        self.metrics["财务费用率"] = self._safe_divide(
            self.metrics["财务费用"],
            self.metrics["营业收入"]
        )

    def _research_and_development_expense_rate(self) -> None:
        """
        研发费用率 = 研发费用 / 营业收入
        """
        self.metrics["研发费用率"] = self._safe_divide(
            self.metrics["研发费用"],
            self.metrics["营业收入"]
        )

    def _period_expense(self) -> None:
        """
        期间费用 = 销售费用 + 管理费用 + 财务费用 + 研发费用
        """
        self.metrics["期间费用"] = (
                self.metrics["销售费用"]
                + self.metrics["管理费用"]
                + self.metrics["财务费用"]
                + self.metrics["研发费用"]
        )

    def _period_expense_rate(self) -> None:
        """
        期间费用率 = （销售费用 + 管理费用 + 财务费用 + 研发费用） / 营业收入
        """
        self.metrics["期间费用率"] = self._safe_divide(
                (
                        self.metrics["销售费用"]
                        + self.metrics["管理费用"]
                        + self.metrics["财务费用"]
                        + self.metrics["研发费用"]
                ),
                self.metrics["营业收入"]
        )

    @depends_on("销售费用率", "毛利率")
    def _difference_between_gross_margin_and_sales_expense_rate(self) -> None:
        """
        毛销差 = 毛利率 - 销售费用率
        """
        self.metrics["毛销差"] = self.metrics["毛利率"] - self.metrics["销售费用率"]

    def _impairment_loss(self) -> None:
        """
        减值损失 = 资产减值损失 + 信用减值损失
        """
        self.metrics["减值损失"] = self.metrics["资产减值损失"] + self.metrics["信用减值损失"]

    @depends_on("核心利润")
    def _proportion_of_impairment_loss(self) -> None:
        """
        减值损失核心利润占比 = (资产减值损失 + 信用减值损失) / 核心利润
        """
        self.metrics["减值损失核心利润占比"] = self._safe_divide(
            self.metrics["资产减值损失"] + self.metrics["信用减值损失"],
            self.metrics["核心利润"].abs()
        )

    def _proportion_of_asset_impairment_loss(self) -> None:
        """
        资产减值损失净资产占比 = 资产减值损失 / 所有者权益
        """
        self.metrics["资产减值损失净资产占比"] = self._safe_divide(
            self.metrics["资产减值损失"],
            self.metrics["所有者权益"]
        )

    def _proportion_of_credit_impairment_loss(self) -> None:
        """
        信用减值损失净资产占比 = 信用减值损失 / 所有者权益
        """
        self.metrics["信用减值损失净资产占比"] = self._safe_divide(
            self.metrics["信用减值损失"],
            self.metrics["所有者权益"]
        )

    def _income_tax_rate(self) -> None:
        """
        所得税税率 = 所得税 / 利润总额
        """
        self.metrics["所得税税率"] = self._safe_divide(
            self.metrics["所得税"],
            self.metrics["利润总额"]
        )

    def _dividend_payout_rate(self) -> None:
        """
        股利支付率 = 分红总额 / 归属于母公司所有者的净利润
        部分两地上市的企业因分红总额错误会出现计算偏差
        """
        self.metrics["股利支付率"] = (
            self._safe_divide(
                self.bonus_data["dividend"],
                self.metrics["归属于母公司所有者的净利润"]
            )
            .clip(lower=0, upper=1)
            .ffill()
        )

    @depends_on("股利支付率")
    def _earnings_retention_rate(self) -> None:
        """
        利润留存率 = 1 - 股利支付率
        """
        self.metrics["利润留存率"] = 1 - self.metrics["股利支付率"]

    @depends_on("营业净利率", "净经营资产周转率", "利润留存率")
    def _internal_growth_rate(self) -> None:
        """
        X = 预计营业净利率 * 预计净经营资产周转率 * 预计利润留存率
        内含增长率 = X / (1 - X)
        """
        x = (
            self.metrics["营业净利率"]
            * self.metrics["净经营资产周转率"]
            * self.metrics["利润留存率"]
        )
        self.metrics["内含增长率"] = x / (1 - x)

    @depends_on("营业净利率", "净经营资产周转率", "净经营资产权益乘数", "利润留存率")
    def _sustainable_growth_rate(self) -> None:
        """
        X = 营业净利率 * 期末净经营资产周转率 * 期末净经营资产权益乘数 * 本期利润留存率
        可持续增长率 = X / (1 - X)
        """
        x = (
            self.metrics["营业净利率"]
            * self.metrics["净经营资产周转率"]
            * self.metrics["净经营资产权益乘数"]
            * self.metrics["利润留存率"]
        )
        self.metrics["可持续增长率"] = x / (1 - x)

    @depends_on("经营净利润", "净经营性长期资产", "经营性营运资本")
    def _operating_cash_flow(self) -> None:
        """
        经营现金流量 = 债务现金流量 + 股权现金流量 = 经营净利润 - (净经营性长期资产增加值 + 经营性营运资本增加值)
        """
        self.metrics["经营现金流量"] = (
                self.metrics["经营净利润"]
                - (self.metrics["净经营性长期资产"].diff(1) + self.metrics["经营性营运资本"].diff(1))
        )

    @depends_on("实际税率", "净负债")
    def _debt_cash_flow(self) -> None:
        """
        债务现金流量 = (利息费用 + 公允价值变动收益 + 投资收益）*（1 - 所得税税率) - (净负债增加值)
        """
        self.metrics["债务现金流量"] = (
                (self.metrics["利息费用"] + self.metrics["公允价值变动收益"] + self.metrics["投资收益"])
                * (1 - self.metrics["实际税率"])
                - self.metrics["净负债"].diff(1)
        )

    def _equity_cash_flow(self) -> None:
        """
        股权现金流量 = 净利润 - 所有者权益增加值
        """
        self.metrics["股权现金流量"] = self.metrics["净利润"] - self.metrics["所有者权益"].diff(1)

    def _current_ratio(self) -> None:
        """
        流动比率 = 流动资产合计 / 流动负债合计
        """
        self.metrics["流动比率"] = self._safe_divide(
            self.metrics["流动资产合计"],
            self.metrics["流动负债合计"]
        )

    def _quick_ratio(self) -> None:
        """
        速动比率 = （货币资金 + 交易性金融资产 + 应收票据及应收账款 + 其他应收款） / 流动负债合计
        """
        self.metrics["速动比率"] = self._safe_divide(
            (self.metrics["货币资金"] + self.metrics["交易性金融资产"]
             + self.metrics["应收票据及应收账款"] + self.metrics["其他应收款"]),
            self.metrics["流动负债合计"]
        )

    def _cash_ratio(self) -> None:
        """
        现金比率 = 货币资金 / 流动负债合计
        """
        self.metrics["现金比率"] = self._safe_divide(
            self.metrics["货币资金"],
            self.metrics["流动负债合计"]
        )

    def _equity_ratio(self) -> None:
        """
        产权比率 = 负债合计 / 所有者权益
        """
        self.metrics["产权比率"] = self._safe_divide(
            self.metrics["负债合计"],
            self.metrics["所有者权益"]
        )

    def _equity_multiplier(self) -> None:
        """
        权益乘数 = 负债和所有者权益 / 所有者权益
        """
        self.metrics["权益乘数"] = self._safe_divide(
            self.metrics["负债和所有者权益"],
            self.metrics["所有者权益"]
        )

    @depends_on("净经营资产")
    def _net_operating_assets_equity_multiplier(self) -> None:
        """
        净经营资产权益乘数 = 净经营资产 / 所有者权益
        """
        self.metrics["净经营资产权益乘数"] = self._safe_divide(
            self.metrics["净经营资产"],
            self.metrics["所有者权益"]
        )

    @depends_on("净负债")
    def _net_leverage(self) -> None:
        """
        净财务杠杆 = 净负债 / 所有者权益
        """
        self.metrics["净财务杠杆"] = self._safe_divide(
            self.metrics["净负债"],
            self.metrics["所有者权益"]
        )

    def _cash_flow_ratio(self) -> None:
        """
        现金流量比率 = 经营活动产生的现金流量净额 / 流动负债合计
        """
        self.metrics["现金流量比率"] = self._safe_divide(
            self.metrics["经营活动产生的现金流量净额"],
            self.metrics["流动负债合计"]
        )

    def _long_term_capitalization_ratio(self) -> None:
        """
        长期资本化比率 = 非流动负债合计 / (非流动负债合计 + 所有者权益)
        """
        self.metrics["长期资本化比率"] = self._safe_divide(
            self.metrics["非流动负债合计"],
            (self.metrics["非流动负债合计"] + self.metrics["所有者权益"]),
        )

    @depends_on("金融性负债")
    def _debt_capitalization_ratio(self) -> None:
        """
        债务资本化比率 = 金融性负债 / （金融性负债 + 所有者权益）
        """
        self.metrics["债务资本化比率"] = self._safe_divide(
            self.metrics["金融性负债"],
            (self.metrics["金融性负债"] + self.metrics["所有者权益"])
        )

    def _asset_liability_ratio(self) -> None:
        """
        资产负债率 = 负债合计 / 负债和所有者权益
        """
        self.metrics["资产负债率"] = self._safe_divide(
            self.metrics["负债合计"],
            self.metrics["负债和所有者权益"]
        )

    @depends_on("金融性负债")
    def _interest_bearing_debt_ratio(self) -> None:
        """
        有息负债率 = 金融性负债 / 负债和所有者权益
        """
        self.metrics["有息负债率"] = self._safe_divide(
            self.metrics["金融性负债"],
            self.metrics["负债和所有者权益"]
        )

    def _interest_coverage_ratio(self) -> None:
        """
        利息保障倍数 = (利润总额 + 利息费用) / 利息费用
        """
        self.metrics["利息保障倍数"] = self._safe_divide(
            (self.metrics["利润总额"] + self.metrics["利息费用"]),
            self.metrics["利息费用"]
        )

    def _cash_flow_interest_coverage_ratio(self) -> None:
        """
        现金流量利息保障倍数 = 经营活动产生的现金流量净额  / 利息费用
        """
        self.metrics["现金流量利息保障倍数"] = self._safe_divide(
            self.metrics["经营活动产生的现金流量净额"],
            self.metrics["利息费用"]
        )

    @depends_on("经营净利润", "经营性资产")
    def _return_on_operating_assets(self) -> None:
        """
        经营性资产收益率 = 经营净利润 / 经营性资产
        """
        self.metrics["经营性资产收益率"] = self._safe_divide(
            self.metrics["经营净利润"],
            self.metrics["经营性资产"].rolling(min_periods=1, window=2).mean()
        )

    @depends_on("金融净利润", "金融性资产")
    def _return_on_financial_assets(self) -> None:
        """
        金融性资产收益率 = 金融净利润 / 金融性资产
        """
        self.metrics["金融性资产收益率"] = self._safe_divide(
            self.metrics["金融净利润"],
            self.metrics["金融性资产"].rolling(min_periods=1, window=2).mean()
        )

    @depends_on("经营净利率", "净经营资产周转率")
    def _net_profit_margin_on_net_operating_assets(self) -> None:
        """
        -1 净经营资产净利率 = 经营净利润 / 净经营资产
        -2 净经营资产净利率 = 经营净利率 * 净经营资产周转率
        """
        self.metrics["净经营资产净利率"] = self.metrics["经营净利率"] * self.metrics["净经营资产周转率"]

    def _net_profit_margin_on_sales(self) -> None:
        """
        营业净利率 = 净利润 / 营业收入
        """
        self.metrics["营业净利率"] = self._safe_divide(
            self.metrics["净利润"],
            self.metrics["营业收入"]
        )

    @depends_on("经营净利润")
    def _net_profit_margin_on_operating(self) -> None:
        """
        经营净利率 = 经营净利润 / 营业收入
        """
        self.metrics["经营净利率"] = self._safe_divide(
            self.metrics["经营净利润"],
            self.metrics["营业收入"]
        )

    def _net_profit_margin_on_asset(self) -> None:
        """
        总资产净利率 = 净利润 / 负债和所有者权益
        """
        self.metrics["总资产净利率"] = self._safe_divide(
            self.metrics["净利润"],
            self.metrics["负债和所有者权益"].rolling(min_periods=1, window=2).mean()
        )

    def _return_on_equity(self) -> None:
        """
        权益净利率 = 净利润 / 所有者权益
        权益净利率 = 净经营资产净利率 + 杠杆贡献率
        """
        self.metrics["权益净利率"] = self._safe_divide(
            self.metrics["净利润"],
            self.metrics["所有者权益"].rolling(min_periods=1, window=2).mean()
        )

    def _return_on_equity_to_shareholder(self) -> None:
        """
        归母权益净利率 = 归属于母公司所有者的净利润 / 所有者权益
        """
        self.metrics["归母权益净利率"] = self._safe_divide(
            self.metrics["归属于母公司所有者的净利润"],
            self.metrics["所有者权益"].rolling(min_periods=1, window=2).mean()
        )

    @depends_on("核心利润")
    def _core_profit_realization_rate(self) -> None:
        """
        核心利润获现率 = 经营活动产生的现金流量净额 / 核心利润
        """
        self.metrics["核心利润获现率"] = self._safe_divide(
            self.metrics["经营活动产生的现金流量净额"],
            self.metrics["核心利润"]
        )

    @depends_on("核心利润")
    def _core_profit_on_equity(self) -> None:
        """
        核心利润净利率 = 核心利润 / 所有者权益
        """
        self.metrics["核心利润净利率"] = self._safe_divide(
            self.metrics["核心利润"],
            self.metrics["所有者权益"].rolling(min_periods=1, window=2).mean()
        )

    def _cash_to_income_ratio(self) -> None:
        """
        收现比 = 销售商品、提供劳务收到的现金 / 营业收入
        """
        self.metrics["收现比"] = self._safe_divide(
            self.metrics["销售商品、提供劳务收到的现金"],
            self.metrics["营业收入"]
        )

    def _cash_to_net_profit_ratio(self) -> None:
        """
        净现比 = 经营活动产生的现金流量净额 / 净利润
        """
        self.metrics["净现比"] = self._safe_divide(
            self.metrics["经营活动产生的现金流量净额"],
            self.metrics["净利润"]
        )

    @depends_on("实际税率", "金融性负债")
    def _after_tax_interest_rate(self) -> None:
        """
        税后利息率 = 利息费用 * (1 - 所得税税率) / 金融性负债
        """
        self.metrics["税后利息率"] = (
            self._safe_divide(
                self.metrics["利息费用"] * (1 - self.metrics["实际税率"]),
                self.metrics["金融性负债"]
            )
            .clip(lower=0)
            .fillna(0)
        )

    @depends_on("净经营资产净利率", "税后利息率")
    def _operating_difference_rate(self) -> None:
        """
        经营差异率 = 净经营资产净利率 - 税后利息率
        """
        self.metrics["经营差异率"] = self.metrics["净经营资产净利率"] - self.metrics["税后利息率"]

    @depends_on("经营差异率", "净财务杠杆")
    def _leverage_contribution_rate(self) -> None:
        """
        杠杆贡献率 = 经营差异率 * 净财务杠杆
        """
        self.metrics["杠杆贡献率"] = self.metrics["经营差异率"] * self.metrics["净财务杠杆"]

    def _inventory_turnover_days(self) -> None:
        """
        存货周转天数 = 360 / （营业成本 / 存货均值）
        """
        self.metrics["存货周转天数"] = 360 / self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["存货"].rolling(min_periods=1, window=2).mean()
        )

    def _receivable_and_bills_turnover_days(self) -> None:
        """
        应收票据及应收账款周转天数 = 360 / (营业收入 / 平均应收票据及应收账款)
        """
        self.metrics["应收票据及应收账款周转天数"] = 360 / self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["应收票据及应收账款"].rolling(min_periods=1, window=2).mean()
        )

    def _payable_and_bills_turnover_days(self) -> None:
        """
        应付票据及应付账款周转天数 = 360 / （营业成本 / 平均应付票据及应付账款 ）
        """
        self.metrics["应付票据及应付账款周转天数"] = 360 / self._safe_divide(
            self.metrics["营业成本"],
            self.metrics["应付票据及应付账款"].rolling(min_periods=1, window=2).mean()
        )

    @depends_on("应付票据及应付账款周转天数", "应收票据及应收账款周转天数")
    def _upstream_and_downstream_turnover_days(self) -> None:
        """
        上下游占款周转天数 = 应付票据及应付账款周转天数 - 应收票据及应收账款周转天数
        """
        self.metrics["上下游占款周转天数"] = self.metrics["应付票据及应付账款周转天数"] - self.metrics["应收票据及应收账款周转天数"]

    def _fixed_assets_turnover_days(self) -> None:
        """
        固定资产周转天数 = 360 / （营业收入 / 固定资产均值）
        """
        self.metrics["固定资产周转天数"] = 360 / self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["固定资产净额"].rolling(min_periods=1, window=2).mean()
        )

    def _current_asset_turnover_days(self) -> None:
        """
        流动资产周转天数 = 360 / (营业收入 / 流动资产合计)
        """
        self.metrics["流动资产周转天数"] = 360 / self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["流动资产合计"].rolling(min_periods=1, window=2).mean()
        )

    def _non_current_asset_turnover_days(self) -> None:
        """
        非流动资产周转天数 = 营业收入 / 非流动资产合计
        """
        self.metrics["非流动资产周转天数"] = 360 / self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["非流动资产合计"].rolling(min_periods=1, window=2).mean()
        )

    def _total_assets_turnover_days(self) -> None:
        """
        总资产周转天数 = 360 / （营业收入 / 负债和所有者权益均值）
        """
        self.metrics["总资产周转天数"] = 360 / self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["负债和所有者权益"].rolling(min_periods=1, window=2).mean()
        )

    @depends_on("应收票据及应收账款周转天数", "存货周转天数")
    def _operating_cycle(self) -> None:
        """
        营业周期 = 存货周转天数 + 应收票据及应收账款周转天数
        """
        self.metrics["营业周期"] = self.metrics["应收票据及应收账款周转天数"] + self.metrics["存货周转天数"]

    @depends_on("应付票据及应付账款周转天数", "应收票据及应收账款周转天数", "存货周转天数")
    def _cash_conversion_cycle(self) -> None:
        """
        现金转换周期 = 应收票据及应收账款周转天数 + 存货周转天数 - 应付票据及应付账款周转天数
        """
        self.metrics["现金转换周期"] = (
                self.metrics["应收票据及应收账款周转天数"]
                + self.metrics["存货周转天数"]
                - self.metrics["应付票据及应付账款周转天数"]
        )

    def _fixed_assets_turnover_rate(self) -> None:
        """
        固定资产周转率 = 营业收入 / 固定资产均值
        """
        self.metrics["固定资产周转率"] = self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["固定资产净额"].rolling(window=2, min_periods=2).mean()
        )

    def _current_asset_turnover_rate(self) -> None:
        """
        流动资产周转率 = 营业收入 / 流动资产合计
        """
        self.metrics["流动资产周转率"] = self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["流动资产合计"].rolling(window=2, min_periods=2).mean()
        )

    def _non_current_asset_turnover_rate(self) -> None:
        """
        非流动资产周转率 = 营业收入 / 非流动资产合计
        """
        self.metrics["非流动资产周转率"] = self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["非流动资产合计"].rolling(window=2, min_periods=2).mean()
        )

    @depends_on("经营性营运资本")
    def _working_capital_turnover_rate(self) -> None:
        """
        营运资本周转率 = 营业收入 / 经营性营运资本
        """
        self.metrics["营运资本周转率"] = self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["经营性营运资本"].rolling(window=2, min_periods=2).mean()
        )

    @depends_on("净经营资产")
    def _net_operating_assets_turnover_rate(self) -> None:
        """
        净经营资产周转率 = 营业收入 / 净经营资产
        """
        self.metrics["净经营资产周转率"] = self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["净经营资产"].rolling(window=2, min_periods=2).mean()
        )

    def _total_assets_turnover_rate(self) -> None:
        """
        总资产周转率 = 营业收入 / 负债和所有者权益均值
        """
        self.metrics["总资产周转率"] = self._safe_divide(
            self.metrics["营业收入"],
            self.metrics["负债和所有者权益"].rolling(window=2, min_periods=2).mean()
        )

    @depends_on("经营性营运资本")
    def _operating_working_capital_required_for_unit_revenue(self) -> None:
        """
        单位营收所需的经营性营运资本 = 经营性营运资本 / 营业收入
        """
        self.metrics["单位营收所需的经营性营运资本"] = self._safe_divide(
            self.metrics["经营性营运资本"].rolling(min_periods=1, window=2).mean(),
            self.metrics["营业收入"]
        )

    def _accounts_payable_ratio(self) -> None:
        """
        应付款比率 = Δ应付票据及应付账款 / 营业成本
        """
        self.metrics["应付款比率"] = self._safe_divide(
            self.metrics["应付票据及应付账款"].diff(1),
            self.metrics["营业成本"]
        )

    def _advance_payment_ratio(self) -> None:
        """
        预收款比率 = Δ预收款项 / 营业收入
        """
        self.metrics["预收款比率"] = self._safe_divide(
            self.metrics["预收款项"].diff(1),
            self.metrics["营业收入"]
        )

    def _upstream_and_downstream_fund_occupation(self) -> None:
        """
        上下游资金占用 = (Δ应付票据及应付账款 + Δ预收款项 - Δ应收票据及应收账款 - Δ预付款项) / 营业收入
        """
        self.metrics["上下游资金占用"] = self._safe_divide(
            (
                    self.metrics["应付票据及应付账款"].diff(1)
                    + self.metrics["预收款项"].diff(1)
                    - self.metrics["应收票据及应收账款"].diff(1)
                    - self.metrics["预付款项"].diff(1)
            ),
            self.metrics["营业收入"]
        )

    def _original_value_of_fixed_assets(self) -> None:
        """
        固定资产原值 = 固定资产净额 + 固定资产和投资性房地产折旧累加值
        """
        self.metrics["固定资产原值"] = self.metrics["固定资产净额"] + self.metrics["固定资产和投资性房地产折旧"].cumsum()

    def _driving_force_of_net_fixed_asset_value(self) -> None:
        """
        固定资产净值推动力 = Δ营业收入 / Δ固定资产净额
        """
        window = {
            "quarter": 2,
        }.get(self.cycle, 1)

        self.metrics["固定资产净值推动力"] = self._safe_divide(
            self.metrics["营业收入"].diff(1),
            self.metrics["固定资产净额"].diff(1).shift(window)
        )

    @depends_on("固定资产原值")
    def _driving_force_of_original_fixed_assets_value(self) -> None:
        """
        固定资产原值推动力 = Δ营业收入 / Δ固定资产原值
        """
        window = {
            "quarter": 2,
        }.get(self.cycle, 1)

        self.metrics["固定资产原值推动力"] = self._safe_divide(
            self.metrics["营业收入"].diff(1),
            self.metrics["固定资产原值"].diff(1).shift(window)
        )

    def _expansion_multiple(self) -> None:
        """
        扩张倍数 = 购建固定资产、无形资产和其他长期资产支付的现金 / (固定资产净额 + 在建工程 + 无形资产)
        """
        self.metrics["扩张倍数"] = self._safe_divide(
            self.metrics["购建固定资产、无形资产和其他长期资产支付的现金"],
            (
                    self.metrics["固定资产净额"]
                    + self.metrics["在建工程"]
                    + self.metrics["无形资产"]
            ).rolling(min_periods=1, window=2).mean()
        )

    @depends_on("金融性负债")
    def shrinkage_multiple(self) -> None:
        """
        收缩倍数 = 偿还债务支付的现金 / 金融性负债
        """
        self.metrics["收缩倍数"] = self._safe_divide(
            self.metrics["偿还债务支付的现金"],
            self.metrics["金融性负债"].rolling(min_periods=1, window=2).mean()
        )

    def _1st_difference_of_revenue(self) -> None:
        """
        营业收入的一阶差分 = 当期营收 - 上一期营收
        """
        self.metrics["营业收入(一阶差分)"] = self.metrics["营业总收入"].diff(1)

    def _inventory_investment(self) -> None:
        """
        存货投资 = 当期存货 - 上一期存货
        """
        self.metrics["存货投资"] = self.metrics["存货"].diff(1)

    def _earnings_before_interest_and_tax(self) -> None:
        """
        息税前利润 = 营业利润 + 利息费用
        """
        self.metrics["息税前利润"] = self.metrics["营业利润"] + self.metrics["利息费用"]

    @depends_on("息税前利润")
    def _earnings_before_interest_and_tax_margin(self) -> None:
        """
        息税前利润 = 营业利润 + 利息费用
        """
        self.metrics["息税前利润率"] = self._safe_divide(
            self.metrics["息税前利润"],
            self.metrics["营业收入"]
        )

    @depends_on("息税前利润", "营运资本")
    def _return_on_invested_capital(self) -> None:
        """
        资本回报率 = 息税前利润 / (净营运资本 + 净固定资产) = 息税前利润 / (营运资本 + 固定资产净额)
        """
        self.metrics["资本回报率"] = self._safe_divide(
            self.metrics["息税前利润"],
            (self.metrics["营运资本"] + self.metrics["固定资产净额"]).rolling(min_periods=1, window=2).mean()
        )

    @depends_on("息税前利润", "超额现金")
    def _return_on_invested_capital1(self) -> None:
        """
        资本回报率 = 息税前利润 / (净营运资本 + 净固定资产) = 息税前利润 / (营运资本 + 固定资产净额)
                                                     = 息税前利润 / (股东权益 + 有息负债 − 超额现金)
        """
        self.metrics["资本回报率1"] = self._safe_divide(
            self.metrics["息税前利润"],
            (
                    self.metrics["所有者权益"]
                    + self.metrics["金融性负债"]
                    - self.metrics["超额现金"]
            ).rolling(min_periods=1, window=2).mean()
        )

    @depends_on("毛利率")
    def _beneish_m_score(self) -> None:
        """
        监测财务造假，M-score > -1.78，有财务造假嫌疑
        M-score = –4.84 +0.92 DSRI + 0.528GMI + 0.404AQI + 0.892SGI + 0.115DEPI–0.172SGAI + 4.67Accruals–0.327LVGI
        DSRI = (Net Receivables t/ Sales t) /（Net Receivables t-1/ Sales t-1)
             = （应收账款净额 / 营业收入） --> 应收票据及应收账款、营业收入
        GMI = [Gross profit t-1 / Sales t-1] / [Gross profit t / Sales t]
            =  【毛利 / 营业收入】 --> 营业收入、营业成本
        AQI = [1 – (Current Assetst + PP&Et) / Total Assetst] / [1 – ((Current Assets t-1+ PP&E t-1)/Total Assets t-1)]
            = 【1 - （流动资产 + 固定资产） / 总资产 】 --> 流动资产、固定资产、总资产
        SGI = Salest / Salest-1
            = 营业收入 --> 营业收入
        DEPI = (Depreciation t-1/ (PP&E t-1 + Depreciation t-1)) / (Depreciation t / (PP&E t + Depreciation t))
             = （折旧 / （固定资产 + 折旧）） --> 固定资产和投资性房地产折旧、固定资产
        SGAI = (SG&A Expense t / Sales t) / (SG&A Expense t-1/ Sales t-1)
             = （SG&A / 营业收入） --> 管理费用、销售费用、营业收入
        LVGI = [(Current Liabilities t + Total Long Term Debt t) / Total Assets t]
        / [(Current Liabilities t-1 + Total Long Term Debt t-1) / Total Assets t-1]
             = 【（流动负债 + 长期负债） / 总资产 】 --> 流动负债、长期借款、应付债券、租赁负债
        Accruals = （（流动资产净额 - 货币资金净额） - （流动负债净额 - 一年内到期长期负债净额 - 应交税费净额） -  折旧） / 总资产
            --> 流动资产、货币资金、流动负债、一年内到期的非流动负债、减：所得税费用、折旧、总资产
            --> 应交税费 = 平均税率 * 营业收入
            --> 平均税率 = mean（所得税费用 / 营业收入）
        """
        # 数据截取（该指标仅适用于年度数据）
        df = self.metrics.filter(regex="-12-31", axis=0).copy()

        model = pd.DataFrame()
        # DSRI
        model["DSRI"] = (
                (df["应收票据及应收账款"] / df["营业收入"])
                / (df["应收票据及应收账款"].shift(1) / df["营业收入"].shift(1))
        )

        # GMI
        model["GMI"] = df["毛利率"].shift(1) / df["毛利率"]

        # AQI
        model["AQI"] = (
                (1 - (df["流动资产合计"] + df["固定资产净额"]) / df["负债和所有者权益"])
            / (1 - (df["流动资产合计"].shift(1) + df["固定资产净额"].shift(1)) / df["负债和所有者权益"].shift(1))
        )

        # SGI
        model["SGI"] = df["营业收入"] / df["营业收入"].shift(1)

        # DEPI
        dep_shift = df["固定资产折旧、油气资产折耗、生产性生物资产折旧"].shift(1)
        ppe_shift = df["固定资产净额"].shift(1)
        model["DEPI"] = (
                (dep_shift / (ppe_shift + dep_shift))
            / (df["固定资产折旧、油气资产折耗、生产性生物资产折旧"] / (df["固定资产净额"] + df["固定资产折旧、油气资产折耗、生产性生物资产折旧"]))
        )

        # SGAI
        model["SGAI"] = (
                ((df["销售费用"] + df["管理费用"]) / df["营业收入"])
            / ((df["销售费用"].shift(1) + df["管理费用"].shift(1)) / df["营业收入"].shift(1))
        )

        # LVGI
        model["LVGI"] = (
                ((df["流动负债合计"] + df["长期借款"] + df["应付债券"] + df["租赁负债"]) / df["负债和所有者权益"])
            / ((df["流动负债合计"].shift(1) + df["长期借款"].shift(1) + df["应付债券"].shift(1) + df["租赁负债"].shift(1)) / df["负债和所有者权益"].shift(1))
        )

        # Accruals
        model["Accruals"] = (
                (
                    (df["流动资产合计"].diff(1) - df["货币资金"].diff(1))
                    - (df["流动负债合计"].diff(1) - df["一年内到期的非流动负债"].diff(1) - df["应交税费"].diff(1))
                    - df["固定资产折旧、油气资产折耗、生产性生物资产折旧"].diff(1)
                )
                / df["资产总计"]
        )
        
        # 处理缺失值
        model.ffill(inplace=True)

        # 计算M-Score
        score = -4.84 + 0.92 * model["DSRI"] + 0.528 * model["GMI"] + 0.404 * model["AQI"] + 0.892 * model["SGI"] \
                + 0.115 * model["DEPI"] - 0.172 * model["SGAI"] + 4.679 * model["Accruals"] - 0.327 * model["LVGI"]

        self.metrics["beneish_m_score"] = score

    def _depr_and_amor(self):
        """
        折旧与摊销 = 固定资产和投资性房地产折旧 + 无形资产摊销 + 长期待摊费用摊销
        """
        self.metrics["折旧与摊销"] = (
                self.metrics["固定资产和投资性房地产折旧"]
                + self.metrics["无形资产摊销"]
                + self.metrics["长期待摊费用摊销"]
        )

    @depends_on("息税前利润", "实际税率", "折旧与摊销", "营运资本")
    def _free_cash_flows_for_firm(self) -> None:
        """
        -1 整体估值（适用 DCF 模型）
        企业自由现金流 = 息税前利润 * （1 - 实际税率） + 折旧、摊销 - 营运资本变动 - 资本性支出（CAPEX）
        实际税率 = 所得税费用 / 利润总额
        折旧、摊销 = 固定资产折旧、油气资产折耗、生产性物资折旧+无形资产摊销+长期待摊费用摊销
        CAPEX = 购建固定资产、无形资产和其他长期资产支付的现金
        """
        self.metrics["企业自由现金流"] = (
            self.metrics["息税前利润"]
            * (1 - self.metrics["实际税率"])
            + self.metrics["折旧与摊销"]
            - self.metrics["营运资本"].diff(1)
            - self.metrics["购建固定资产、无形资产和其他长期资产支付的现金"]
        )

    @depends_on("折旧与摊销", "营运资本")
    def _free_cash_flow_for_equity(self) -> None:
        """
        -1 股权估值（使用 股息贴现模型 或 股权自由现金流折现）
        股权自由现金流 = 净利润 + 折旧、摊销 - 营运资本变动 - 资本性支出（CAPEX） + （新增借款 - 偿还债务）
        新增借款 = 取得借款收到的现金（现金流量表）
        偿还债务 = 偿还债务支付的现金（现金流量表）
        """
        self.metrics["股权自由现金流"] = (
            self.metrics["净利润"]
            + self.metrics["折旧与摊销"]
            - self.metrics["营运资本"].diff(1)
            - self.metrics["购建固定资产、无形资产和其他长期资产支付的现金"]
            + (self.metrics["取得借款收到的现金"] - self.metrics["偿还债务支付的现金"])
        )

    @depends_on("实际税率")
    def interest_tax_shield(self):
        """
        利息税盾 = 利息费用 * 实际税率
        """
        self.metrics["利息税盾"] = self.metrics["利息费用"] * self.metrics["实际税率"]

    @depends_on("企业自由现金流", "利息税盾")
    def _unlevered_free_cash_flow(self) -> None:
        """
        -1 并购重组（评估核心资产价值）； -2 行业对比（比较经营效率）
        无杠杆自由现金流 = 企业自由现金流 + 利息税盾
        利息税盾 = 利息支出 * 实际税率
        """
        self.metrics["无杠杆自由现金流"] = self.metrics["企业自由现金流"] + self.metrics["利息税盾"]

    @depends_on("折旧与摊销", "营运资本", "实际税率")
    def _levered_free_cash_flow(self) -> None:
        """
        -1 偿债能力分析（利息覆盖率 = LFCF / 利息支出）
        杠杆自由现金流 = 净利润 + 折旧、摊销 − 营运资本变动 - 资本性支出（CAPEX）+ 税后利息费用
        税后利息费用 = 利息费用 * （1 - 实际税率）
        """
        self.metrics["杠杆自由现金流"] = (
            self.metrics["净利润"]
            + self.metrics["折旧与摊销"]
            - self.metrics["营运资本"].diff(1)
            - self.metrics["购建固定资产、无形资产和其他长期资产支付的现金"]
            + (self.metrics["利息费用"] * (1 - self.metrics["实际税率"]))
        )

    def _average_income_tax_rate(self) -> None:
        """
        平均所得税税率 = (所得税 / 利润总额) 的均值
        算法逻辑：
            -1 仅当利润总额>0且所得税>0时计算有效税率
            -2 税率超过1的视为异常值(理论上所得税不应超过利润总额)
            -3 使用MAD法去极值(比标准差法更鲁棒)
            -4 添加平滑处理避免0值问题
        """
        # 获取关键数据列
        profit = self.metrics["利润总额"]
        tax = self.metrics["所得税"]

        # 创建有效数据掩码
        mask = (profit > 0) & (tax > 0) & (tax < profit)
        valid_profit = profit[mask]
        valid_tax = tax[mask]

        # 处理无有效数据情况
        if valid_profit.empty:
            self.metrics["平均所得税税率"] = 0.0
            return

        # 计算原始税率
        tax_rates = valid_tax / valid_profit
        # 去极值
        cleaned_rates = self.de_extreme_method(tax_rates)
        # 计算加权平均（按利润总额加权）
        weighted_avg = np.average(cleaned_rates, weights=valid_profit[cleaned_rates.index])
        # 添加平滑处理防止0值
        self.metrics["平均所得税税率"] = weighted_avg if weighted_avg > 1e-4 else 0.0

    def _effective_tax_rate(self) -> None:
        """
        实际税率 = 所得税 / 利润总额
        """
        # 处理分母有效性（利润总额 <=0 时返回NaN）
        self.metrics["实际税率"] = np.where(
            self.metrics["利润总额"] > 0,
            self.metrics["所得税"] / self.metrics["利润总额"],
            np.nan
        )

        # 异常值处理（税率超过100%或为负）
        self.metrics["实际税率"] = self.metrics["实际税率"].apply(
            lambda x: x if (0 <= x <= 1) else np.nan
        )

        # 向前填充
        self.metrics["实际税率"] = self.metrics["实际税率"].ffill()
        # 用0填充
        self.metrics["实际税率"] = self.metrics["实际税率"].fillna(0)
