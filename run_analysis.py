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
        start_date="2015-12-31",
        end_date="2025-03-27",
        financial_end_date="2024-12-31",
        storage_dir_name="益生股份24Y",
        target_info="002458"
    )
    analyzer.run()


# ---------------------------------------------------
def normal_analysis():
    """分析（宏观、中观、微观）"""
    analyzer = Analyzer(
        index_code="000300",
        params=Normal(),

        dimension="micro",
        cycle="month",
        financial_cycle="quarter",
        start_date="2021-09-30",
        end_date="2025-03-23",
        financial_end_date="2024-09-30",
        storage_dir_name="二轮选拔24Q3",
        # "三元正极": "自定义一", "磷酸铁锂正极": "自定义一", "前驱体": "自定义一",
        #                      "负极": "自定义", "电解液": "自定义", "隔膜": "自定义"
        target_info=["300919", "300073", "603659", "300037", "002812"],
        class_level="自定义一",
        weight_name="市值",
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
        dimension="micro",
        cycle="month",
        financial_cycle="quarter",
        start_date="2000-09-30",
        end_date="2025-03-24",
        financial_end_date="2024-09-30",
        weight_name="市值",
        storage_dir_name="202503M",
        target_info={"全部": "三级行业"},
        class_level="三级行业",
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
        dimension="micro",
        cycle="month",
        financial_cycle="quarter",
        start_date="2000-09-30",
        end_date="2025-03-24",
        financial_end_date="2024-09-30",
        weight_name="市值",
        storage_dir_name="202503M",
        target_info={"全部": "三级行业"},
        class_level="三级行业",
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
    individual_analysis()
    # normal_analysis()
    # quant_analysis()
    # factor_analysis()
    # inventory_cycle_analysis()
