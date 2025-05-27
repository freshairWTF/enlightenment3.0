from analysis.analysis_service import Analyzer
from constant.analysis_setting import Individual, Normal, Factor, ModelFactor, InventoryCycle
from quant.data_convert import DataConvert


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
        start_date="2000-03-30",
        financial_end_date="2025-03-31",
        end_date="2025-05-12",
        storage_dir_name="公司治理指标测试25Q1",
        target_info="603896",
        debug=True
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
        end_date="2025-05-08",

        storage_dir_name="风电零部件25Q1",
        target_info={"风电零部件": "三级行业"},
        draw_filter=True,
        debug=False
    )
    analyzer.run()


# ---------------------------------------------------
def factor_analysis():
    """单因子分析 -> 生成分析因子"""
    storage_dir_name = "公司治理指标"
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
        financial_cycle="quarter",
        start_date="2000-03-31",
        financial_end_date="2024-09-30",
        end_date="2025-05-09",

        storage_dir_name=storage_dir_name,
        target_info={"全部": "三级行业"},

        index_code="000300",
        processes_nums=10,
        debug=False
    )
    analyzer.run()

    convert = DataConvert(
        source_dir=storage_dir_name,
        storage_dir=storage_dir_name
    )
    convert.run()

# ---------------------------------------------------
def model_factor_analysis():
    """
    模型分析 -> 生成因子库中的因子
    注：更新指数数据 -> 更新交易日历 -> 最新数据
    """
    storage_dir_name = "20250504-WEEK"
    analyzer = Analyzer(
        quant=True,
        code_range="Total_A",
        trade_status_filter=True,
        params=ModelFactor(),
        kline_adjust="backward_adjusted",

        # 运存不足，无法满足 day 的运存消耗
        dimension="micro",
        class_level="三级行业",
        weight_name="市值",

        cycle="week",
        financial_cycle="quarter",
        start_date="2015-01-31",
        financial_end_date="2025-03-31",
        end_date="2025-05-23",

        storage_dir_name=storage_dir_name,
        target_info={"全部": "三级行业"},

        index_code="000300",
        processes_nums=10,
        debug=False
    )
    analyzer.run()

    convert = DataConvert(
        source_dir=storage_dir_name,
        storage_dir=storage_dir_name
    )
    convert.run()

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

        start_date="2021-03-31",
        end_date="2025-04-30",
        financial_end_date="2025-03-31",

        storage_dir_name="三级行业-库存周期25Q1",
        target_info={"全部": "三级行业"},
        debug=False
    )
    analyzer.run()


# ---------------------------------------------------
if __name__ == "__main__":
    # individual_analysis()
    # normal_analysis()
    factor_analysis()
    # model_factor_analysis()
    # inventory_cycle_analysis()

    """
    行尾金融因子  阳线天数、阴线天数
    技术面因子  真实波幅的波动率
    市场情绪因子  市场最高连扳数
    """