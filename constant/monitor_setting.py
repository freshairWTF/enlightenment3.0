"""
监控常量
"""

from draw_specs import BasicChartSpecs, QuadrantsChartSpecs


#####################################################
class Factor:

    def __init__(self):
        # 财务模块
        self.finance = self.Finance()
        # 估值模块
        self.valuation = self.Valuation()
        # 量价模块
        self.kline = self.Kline()
        # 统计模块
        self.stat = self.Statistics()
        # 可视化模块
        self.visualization = self.Visualization()

    class Finance:
        def __init__(self):
            """
            盈利能力：衡量企业赚取利润的能力。
            运营能力：反映企业资产运营效率和管理水平。
            偿债能力：评估企业偿还债务的能力。
            成长能力：衡量企业未来发展潜力和增长能力。
            现金流量：分析企业现金流入和流出的情况。
            资本结构：反映企业资金来源和构成。
            费用控制：评估企业对各项费用的控制能力。
            资产质量：衡量企业资产的安全性和价值。
            税务管理：反映企业税务负担和管理效率。
            其他：未明确归类的指标，通常为基础数据或辅助分析指标。
            """
            # 基础报表
            self.basic_reports = {
                "资产负债表": [
                    "所有者权益",
                ],
                "利润表": [
                ],
                "现金流量表": [
                ]
            }
            # 财务分析
            self.financial_analysis = {
                "盈利能力": [
                    "核心利润"
                ],
                "运营能力": [
                ],
                "偿债能力": [
                ],
                "成长能力": [
                ],
                "现金流量": [
                ],
                "资本结构": [
                ],
                "费用控制": [
                ],
                "资产质量": [
                ],
                "税务管理": [
                ],
                "其他": [
                ]
            }

    class Valuation:
        def __init__(self):
            # 基础指标
            self.basic_metrics = [
                "市值", "对数市值", "市净率", "市销率", "核心利润盈利市值比",
            ]
            # 衍生指标
            self.derived_metrics = [
            ]

    class Kline:
        def __init__(self):
            self.kline = {
            }

    class Statistics:
        def __init__(self):
            self.stats = ["同比", "归一化"]

    class Visualization:
        def __init__(self):
            self.pages_name = [
                "相关性", "比率趋势", "市净率", "对数市值"
            ]
            # 图表配置
            self.pages_config = {
                "相关性": {
                    "_basic_heat_map-1": BasicChartSpecs(
                        title="相关系数矩阵",
                        data_source="corr",
                        column="",
                        date=False
                    ),
                    "_basic_heat_map-2": BasicChartSpecs(
                        title="近期相关系数矩阵",
                        data_source="recent_corr",
                        column="",
                        date=False
                    ),
                    "_basic_line-1": BasicChartSpecs(
                        title="配对相关性",
                        data_source="pairwise_corr",
                        column="",
                        date=False
                    )
                },
                "比率趋势": {
                    "_basic_line-1": BasicChartSpecs(
                        title="估值价差-市净率",
                        data_source="valuation_spread_市净率",
                        column="",
                        date=False
                    ),
                    "_basic_line-2": BasicChartSpecs(
                        title="估值价差-核心利润盈利市值比",
                        data_source="valuation_spread_核心利润盈利市值比",
                        column="",
                        date=False
                    ),
                    "_basic_line-6": BasicChartSpecs(
                        title="估值价差-市销率",
                        data_source="valuation_spread_市销率",
                        column="",
                        date=False
                    ),
                    "_basic_line-3": BasicChartSpecs(
                        title="累计收益率",
                        data_source="cum",
                        column="",
                        date=False
                    ),
                    "_basic_line-4": BasicChartSpecs(
                        title="波动率",
                        data_source="vol",
                        column="",
                        date=False
                    ),
                    "_basic_line-5": BasicChartSpecs(
                        title="换手率",
                        data_source="turn",
                        column="",
                        date=False
                    ),
                },
                "市净率": {
                    "_basic_line-1": BasicChartSpecs(
                        title="估值价差-核心利润盈利市值比",
                        data_source="processed_市净率_valuation_spread_核心利润盈利市值比",
                        column="",
                        date=False
                    ),
                    "_basic_line-5": BasicChartSpecs(
                        title="估值价差-市销率",
                        data_source="processed_市净率_valuation_spread_市销率",
                        column="",
                        date=False
                    ),
                    "_basic_line-2": BasicChartSpecs(
                        title="累计收益率",
                        data_source="processed_市净率_cum",
                        column="",
                        date=False
                    ),
                    "_basic_line-3": BasicChartSpecs(
                        title="波动率",
                        data_source="processed_市净率_vol",
                        column="",
                        date=False
                    ),
                    "_basic_line-4": BasicChartSpecs(
                        title="换手率",
                        data_source="processed_市净率_turn",
                        column="",
                        date=False
                    ),
                },
                "对数市值": {
                    "_basic_line-1": BasicChartSpecs(
                        title="估值价差-市净率",
                        data_source="processed_对数市值_valuation_spread_市净率",
                        column="",
                        date=False
                    ),
                    "_basic_line-5": BasicChartSpecs(
                        title="估值价差-核心利润盈利市值比",
                        data_source="processed_对数市值_valuation_spread_核心利润盈利市值比",
                        column="",
                        date=False
                    ),
                    "_basic_line-6": BasicChartSpecs(
                        title="估值价差-市销率",
                        data_source="processed_对数市值_valuation_spread_市销率",
                        column="",
                        date=False
                    ),
                    "_basic_line-2": BasicChartSpecs(
                        title="累计收益率",
                        data_source="processed_对数市值_cum",
                        column="",
                        date=False
                    ),
                    "_basic_line-3": BasicChartSpecs(
                        title="波动率",
                        data_source="processed_对数市值_vol",
                        column="",
                        date=False
                    ),
                    "_basic_line-4": BasicChartSpecs(
                        title="换手率",
                        data_source="processed_对数市值_turn",
                        column="",
                        date=False
                    ),

                },
            }


#####################################################
class Kline:

    def __init__(self):
        # 量价模块
        self.kline = self.Kline()
        # 可视化模块
        self.visualization = self.Visualization()

    class Kline:
        def __init__(self):
            self.kline = {
                "斜率": [
                    0.024, 0.08, 0.16
                ]
            }

    class Visualization:
        def __init__(self):
            # page名
            self.pages_name = []
            # 图表配置
            self.pages_config: dict[str, BasicChartSpecs | QuadrantsChartSpecs] = {}
