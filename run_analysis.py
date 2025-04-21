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
        financial_cycle="year",
        start_date="2015-03-30",
        financial_end_date="2024-12-31",
        end_date="2025-04-12",
        storage_dir_name="锦泓集团24Y",
        target_info="603518",
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
        start_date="2015-03-31",
        financial_end_date="2024-09-30",
        end_date="2025-04-17",

        storage_dir_name="零食250403W",
        target_info={"零食": "三级行业"},
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
        params=Quant(),

        # 运存不足，无法满足 day 的运存消耗
        dimension="meso",
        class_level="三级行业",
        weight_name="市值",

        cycle="month",
        financial_cycle="quarter",
        start_date="2022-03-31",
        financial_end_date="2024-09-30",
        end_date="2025-04-11",

        storage_dir_name="20250402M估值历史分位数",
        target_info={"全部": "三级行业"},

        index_code="000300",
        processes_nums=10,
        debug=False
    )
    analyzer.run()


# ---------------------------------------------------
def factor_analysis():
    """量化分析 -> 生成分析因子"""
    analyzer = Analyzer(
        quant=True,
        code_range="Total_A",
        params=Factor(),

        # 运存不足，无法满足 day 的运存消耗
        dimension="micro",
        class_level="三级行业",
        weight_name="市值",

        cycle="week",
        financial_cycle="quarter",
        start_date="2000-03-31",
        financial_end_date="2024-09-30",
        end_date="2025-04-03",

        storage_dir_name="20250401W量价",
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
        class_level="一级行业",

        start_date="2021-03-31",
        end_date="2024-09-30",
        financial_end_date="2024-09-30",

        storage_dir_name="库存周期24Q4",
        target_info={"全部": "三级行业"},
    )
    analyzer.run()


# ---------------------------------------------------
if __name__ == "__main__":
    individual_analysis()
    # normal_analysis()
    # quant_analysis()
    # factor_analysis()
    # inventory_cycle_analysis()
