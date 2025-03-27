from analysis.analysis_service_debug import Analyzer
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
        start_date="2000-03-31",
        end_date="2024-09-30",
        storage_dir_name="格力电器",
        target_info="000651",
        dcf_valuation=False
    )
    analyzer.run()


# ---------------------------------------------------
def normal_analysis():
    """分析（宏观、中观、微观）"""
    analyzer = Analyzer(
        index_code="000300",
        params=Normal(),

        dimension="meso",
        cycle="month",
        start_date="2021-03-31",
        end_date="2024-09-30",
        storage_dir_name="稳定性测试",
        target_info=["688711", "688697", "688700"],
        weight_name="等权",
        draw_filter=True,
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
        cycle="month",
        financial_cycle="quarter",
        dimension="micro",
        start_date="2000-03-31",
        end_date="2025-03-11",
        weight_name="等权",
        storage_dir_name="测试",
        target_info={"全部": "一级行业"},
        # target_info=["688711", "688697", "688700"],
        index_code="000300",
        processes_nums=10,
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
        cycle="month",
        financial_cycle="quarter",
        dimension="micro",
        start_date="2000-03-31",
        end_date="2025-03-11",
        weight_name="等权",
        storage_dir_name="测试",
        target_info={"全部": "一级行业"},
        # target_info=["688711", "688697", "688700"],
        index_code="000300",
        processes_nums=10,
    )
    analyzer.run()


# ---------------------------------------------------
def inventory_cycle_analysis():
    """库存周期跟踪"""
    analyzer = Analyzer(
        code_range="A",
        cycle="quarter",
        dimension="meso",
        index_code="000300",
        draw_filter=True,
        params=InventoryCycle(),

        start_date="2021-03-31",
        end_date="2024-09-30",
        weight_name="等权",
        storage_dir_name="稳定性测试",
        target_info=["688711", "688697", "688700"],
    )
    analyzer.run()


if __name__ == "__main__":
    # individual_analysis()
    # normal_analysis()
    quant_analysis()
    # factor_analysis()
    # inventory_cycle_analysis()
