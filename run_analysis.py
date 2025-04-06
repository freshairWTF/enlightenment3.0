from analysis.analysis_service import Analyzer
from constant.analysis_setting import Individual, Normal, Quant, Factor, InventoryCycle


"""
        "三元正极": "自定义一", "磷酸铁锂正极": "自定义一", "前驱体": "自定义一",
                             "负极": "自定义", "电解液": "自定义", "隔膜": "自定义"
"""


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
        start_date="2015-12-31",
        financial_end_date="2024-12-31",
        end_date="2025-03-28",
        storage_dir_name="科思股份24Q3",
        target_info="300856"
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
        start_date="2021-09-30",
        end_date="2025-03-23",
        financial_end_date="2024-09-30",

        storage_dir_name="品牌化妆品24Q3",
        target_info={"品牌化妆品": "三级行业"},
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
        dimension="micro",
        class_level="三级行业",
        weight_name="市值",

        cycle="month",
        financial_cycle="quarter",
        start_date="2000-03-31",
        financial_end_date="2024-09-30",
        end_date="2025-03-31",

        storage_dir_name="202503M财务估值",
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

        cycle="month",
        financial_cycle="quarter",
        start_date="2000-03-31",
        financial_end_date="2024-09-30",
        end_date="2025-03-31",

        storage_dir_name="202503M量价",
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

        start_date="2021-03-31",
        end_date="2024-09-30",
        financial_end_date="2024-09-30",
        weight_name="市值",
        storage_dir_name="稳定性测试",
        target_info=["688711", "688697", "688700"],
        class_level="一级行业",
    )
    analyzer.run()


if __name__ == "__main__":
    # individual_analysis()
    # normal_analysis()
    # quant_analysis()
    factor_analysis()
    # inventory_cycle_analysis()
