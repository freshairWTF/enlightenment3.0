from analysis.analysis_service import Analyzer
from constant.analysis_setting import Individual, Normal, Quant, Factor, InventoryCycle


# ---------------------------------------------------
def individual_analysis():
    """个股分析"""
    analyzer = Analyzer(
        dimension="micro",
        draw_filter=True,
        index_code="000300",
        code_range="Total_A",
        params=Individual(),

        cycle="month",
        financial_cycle="quarter",
        start_date="2020-03-30",
        financial_end_date="2025-03-31",
        end_date="2025-05-03",
        storage_dir_name="新宙邦25Q1",
        target_info="300037",
        debug=False
    )
    analyzer.run()


# ---------------------------------------------------
def normal_analysis():
    """分析（宏观、中观、微观）"""
    analyzer = Analyzer(
        index_code="000300",
        params=Normal(),

        dimension="micro",
        class_level="三级行业",
        weight_name="市值",

        cycle="month",
        financial_cycle="quarter",
        start_date="2021-03-31",
        financial_end_date="2025-03-31",
        end_date="2025-05-02",

        storage_dir_name="电池化学品25Q1",
        target_info={"电池化学品": "三级行业"},
        draw_filter=True,
        debug=False
    )
    analyzer.run()


# ---------------------------------------------------
def quant_analysis():
    """量化分析 -> 生成分析因子"""
    analyzer = Analyzer(
        quant=True,
        code_range="Total_A",
        trade_status_filter=True,
        params=Quant(),

        # 运存不足，无法满足 day 的运存消耗
        dimension="micro",
        class_level="三级行业",
        weight_name="市值",

        cycle="week",
        financial_cycle="quarter",
        start_date="2000-03-31",
        financial_end_date="2024-09-30",
        end_date="2025-04-25",

        storage_dir_name="202504W",
        target_info={"全部": "三级行业"},

        index_code="000300",
        processes_nums=10,
        debug=False
    )
    analyzer.run()


# ---------------------------------------------------
def factor_analysis():
    """
    量化分析 -> 生成因子库中的因子
    注：更新指数数据 -> 更新交易日历 -> 最新数据
    """
    analyzer = Analyzer(
        quant=True,
        code_range="Total_A",
        trade_status_filter=True,
        params=Factor(),

        # 运存不足，无法满足 day 的运存消耗
        dimension="micro",
        class_level="三级行业",
        weight_name="市值",

        cycle="week",
        kline_adjust="split_adjusted",
        financial_cycle="quarter",
        start_date="2000-03-31",
        financial_end_date="2025-03-31",
        end_date="2025-04-30",

        storage_dir_name="20250501-WEEK",
        target_info={"全部": "三级行业"},

        index_code="000300",
        processes_nums=10,
        debug=False
    )
    analyzer.run()


# ---------------------------------------------------
def inventory_cycle_analysis():
    """库存周期跟踪"""
    analyzer = Analyzer(
        code_range="A",
        cycle="quarter",
        financial_cycle="quarter",
        dimension="meso",
        index_code="000300",
        draw_filter=True,
        params=InventoryCycle(),

        weight_name="市值",
        class_level="三级行业",

        start_date="2022-03-31",
        end_date="2025-04-30",
        financial_end_date="2025-03-31",

        storage_dir_name="库存周期25Q1",
        target_info={"全部": "三级行业"},
        debug=False
    )
    analyzer.run()


# ---------------------------------------------------
if __name__ == "__main__":
    # individual_analysis()
    # normal_analysis()
    # quant_analysis()
    factor_analysis()
    # inventory_cycle_analysis()

    # 长期斜率为正，突然杀跌