from analysis.analysis_service import Analyzer
from constant.analysis_setting import Individual, Normal, Factor, ModelFactor, InventoryCycle
from data_convert import DataConvert


# ---------------------------------------------------
def individual_analysis():
    """个股分析"""
    analyzer = Analyzer(
        dimension="micro",
        draw_filter=True,
        index_code="000300",
        code_range="Total_A",
        params=Individual(),

        cycle="week",
        financial_cycle="quarter",
        start_date="2000-03-31",
        financial_end_date="2022-12-31",
        end_date="2022-12-31",
        storage_dir_name="公司治理指标测试25Q1",
        target_info="301231",
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

        storage_dir_name="汽车零部件25Q1",
        target_info={"汽车零部件": "二级行业"},
        draw_filter=True,
        debug=False
    )
    analyzer.run()

# ---------------------------------------------------
def factor_analysis():
    """单因子分析 -> 生成分析因子"""
    storage_dir_name = "带波动率过滤的累计收益率"
    analyzer = Analyzer(
        quant=True,
        code_range="Total_A",
        trade_status_filter=True,
        storage_dir_name=storage_dir_name,
        params=Factor(),

        # 运存不足，无法满足 day 的运存消耗
        dimension="micro",
        class_level="三级行业",
        weight_name="市值",

        cycle="day",
        financial_cycle="quarter",
        start_date="2010-03-31",
        financial_end_date="2022-12-31",
        end_date="2022-12-31",

        target_info={"全部": "三级行业"},
        index_code="000300",

        processes_nums=10,
        debug=False
    )
    analyzer.run()

# ---------------------------------------------------
def model_factor_analysis():
    """
    模型分析 -> 生成因子库中的因子
    注：更新指数数据 -> 更新交易日历 -> 最新数据
    """
    storage_dir_name = "模型因子训练集"
    analyzer = Analyzer(
        quant=True,
        code_range="Total_A",
        trade_status_filter=True,
        params=ModelFactor(),
        storage_dir_name=storage_dir_name,
        kline_adjust="backward_adjusted",

        # 运存不足，无法满足 day 的运存消耗
        dimension="micro",
        class_level="三级行业",
        weight_name="市值",

        cycle="week",
        financial_cycle="quarter",
        start_date="2000-01-01",
        financial_end_date="2022-12-31",
        end_date="2025-06-13",

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

        start_date="2021-03-31",
        end_date="2025-04-30",
        financial_end_date="2025-03-31",

        storage_dir_name="三级行业-库存周期25Q1",
        target_info={"全部": "三级行业"},
        debug=False
    )
    analyzer.run()

# ---------------------------------------------------
def data_convert():
    """分析因子 -> 量化因子"""
    convert = DataConvert(
        source_dir="带波动率过滤的累计收益率",
        storage_dir="带波动率过滤的累计收益率"
    )
    convert.run()


# ---------------------------------------------------
if __name__ == "__main__":
    # individual_analysis()
    # normal_analysis()
    # factor_analysis()
    # model_factor_analysis()
    # inventory_cycle_analysis()
    data_convert()
