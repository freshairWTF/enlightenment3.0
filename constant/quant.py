"""
量化常量
"""

from draw_specs import BasicChartSpecs


# 负值单列（-1 用于单因子分析 -2 部分想探究负值影响的指标，一般使用倒数替代）
NEGATIVE_SINGLE_COLUMN = [
    '市净率', '市销率', '核心利润市盈率'
]

# 禁止中性化的因子（用于单因子分析）
PROHIBIT_MV_NEUTRAL = ["市值", "对数市值"]

# 行业映射表
CLASS_MAPPING = {
    "1": "一级行业",
    "2": "二级行业",
    "3": "三级行业"
}

# 周期年化
ANNUALIZED_DAYS = {
    "day": 252,
    "week": 52,
    "month": 12,
    "quarter": 4,
    "half": 2,
    "year": 1
}


#####################################################
class Factor:

    def __init__(self):
        # 指数量价模块
        self.kline = self.Kline()
        # 可视化模块
        self.visualization = self.Visualization()

    class Kline:
        def __init__(self):
            self.kline = {
                "累加收益率": [
                    0.25
                ],
                "收益率标准差": [
                    0.25
                ],
                "斜率": [
                    0.25
                ],
                "收盘价均线": [
                    0.25, 1
                ]
            }

    class Visualization:
        def __init__(self):
            self.pages_name = [
                "因子分析"
            ]
            # 图表配置
            self.pages_config = {
                "因子分析": {
                    "_basic_bar-1": BasicChartSpecs(
                        title="因子覆盖度",
                        data_source="coverage",
                        column="",
                        date=False
                    ),
                    "_basic_table-1": BasicChartSpecs(
                        title="因子描述统计",
                        data_source="desc_stats",
                        column="",
                        date=False
                    ),
                    "_basic_table-2": BasicChartSpecs(
                        title="IC统计",
                        data_source="ic_stats",
                        column="",
                        date=False
                    ),
                    "_basic_line-1": BasicChartSpecs(
                        title="IC累加",
                        data_source="ic_cumsum",
                        column="",
                        date=False
                    ),
                    "_basic_bar-2": BasicChartSpecs(
                        title="IC衰退",
                        data_source="ic_decay",
                        column="",
                        date=False
                    ),
                    "_basic_table-7": BasicChartSpecs(
                        title="分市场IC",
                        data_source="different_market_result",
                        column="",
                        date=False
                    ),
                    "_basic_table-5": BasicChartSpecs(
                        title="基础信息",
                        data_source="basic_stats",
                        column="",
                        date=False
                    ),
                    "_basic_table-3": BasicChartSpecs(
                        title="收益指标",
                        data_source="returns_stats",
                        column="",
                        date=False
                    ),
                    "_basic_line-2": BasicChartSpecs(
                        title="收益累计",
                        data_source="cum_returns",
                        column="",
                        date=False
                    ),
                    "_basic_table-4": BasicChartSpecs(
                        title="基础信息",
                        data_source="mw_basic_stats",
                        column="",
                        date=False
                    ),
                    "_basic_table-6": BasicChartSpecs(
                        title="收益指标",
                        data_source="mw_returns_stats",
                        column="",
                        date=False
                    ),
                    "_basic_line-3": BasicChartSpecs(
                        title="收益累计",
                        data_source="mw_cum_returns",
                        column="",
                        date=False
                    )
                }
            }


#####################################################
class ModelVisualization:
    def __init__(self):
        self.pages_name = [
            "模型分析", "换手率估计_模型分析", "1%TS_模型分析", "3%TS_模型分析", "5%TS_模型分析",
        ]
        # 图表配置
        self.pages_config = {
            "模型分析": {
                "_basic_bar-1": BasicChartSpecs(
                    title="因子覆盖度",
                    data_source="coverage",
                    column="",
                    date=False
                ),
                "_basic_table-7": BasicChartSpecs(
                    title="模型评估",
                    data_source="模型评估指标",
                    column="",
                    date=False
                ),
                "_basic_table-2": BasicChartSpecs(
                    title="IC统计",
                    data_source="ic_stats",
                    column="",
                    date=False
                ),
                "_basic_line-1": BasicChartSpecs(
                    title="IC累加",
                    data_source="ic_cumsum",
                    column="",
                    date=False
                ),
                "_basic_table-3": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.0_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-2": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.0_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-6": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.0_mw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-3": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.0_mw_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-9": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.0_pw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-5": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.0_pw_cum_returns",
                    column="",
                    date=False
                )
            },
            "换手率估计_模型分析": {
                "_basic_bar-1": BasicChartSpecs(
                    title="因子覆盖度",
                    data_source="coverage",
                    column="",
                    date=False
                ),
                "_basic_table-7": BasicChartSpecs(
                    title="模型评估",
                    data_source="模型评估指标",
                    column="",
                    date=False
                ),
                "_basic_table-2": BasicChartSpecs(
                    title="IC统计",
                    data_source="ic_stats",
                    column="",
                    date=False
                ),
                "_basic_line-1": BasicChartSpecs(
                    title="IC累加",
                    data_source="ic_cumsum",
                    column="",
                    date=False
                ),
                "_basic_table-3": BasicChartSpecs(
                    title="收益指标",
                    data_source="换手率估计_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-2": BasicChartSpecs(
                    title="收益累计",
                    data_source="换手率估计_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-6": BasicChartSpecs(
                    title="收益指标",
                    data_source="换手率估计_mw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-3": BasicChartSpecs(
                    title="收益累计",
                    data_source="换手率估计_mw_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-9": BasicChartSpecs(
                    title="收益指标",
                    data_source="换手率估计_pw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-5": BasicChartSpecs(
                    title="收益累计",
                    data_source="换手率估计_pw_cum_returns",
                    column="",
                    date=False
                )
            },
            "1%TS_模型分析": {
                "_basic_table-3": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.01_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-2": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.01_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-6": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.01_mw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-3": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.01_mw_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-9": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.01_pw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-5": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.01_pw_cum_returns",
                    column="",
                    date=False
                )
            },
            "3%TS_模型分析": {
                "_basic_table-3": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.03_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-2": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.03_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-6": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.03_mw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-3": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.03_mw_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-9": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.03_pw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-5": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.03_pw_cum_returns",
                    column="",
                    date=False
                )
            },
            "5%TS_模型分析": {
                "_basic_table-3": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.05_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-2": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.05_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-6": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.05_mw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-3": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.05_mw_cum_returns",
                    column="",
                    date=False
                ),
                "_basic_table-9": BasicChartSpecs(
                    title="收益指标",
                    data_source="0.05_pw_returns_stats",
                    column="",
                    date=False
                ),
                "_basic_line-5": BasicChartSpecs(
                    title="收益累计",
                    data_source="0.05_pw_cum_returns",
                    column="",
                    date=False
                )
            },
        }


