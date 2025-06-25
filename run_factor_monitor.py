from analysis.analysis_service import Analyzer
from data_convert import DataConvert
from quant.factor_monitor_service import FactorMonitor
from constant.monitor_setting import Factor
from constant.factor_library import FACTOR_LIBRARY


# ---------------------------------------------------
def quant_analysis(
        start_date_: str,
        end_date_: str
) -> None:
    """量化分析 -> 生成分析因子"""
    analyzer = Analyzer(
        quant=True,
        code_range="A",
        params=Factor(),
        dimension="micro",
        target_info={"全部": "一级行业"},
        index_code="000300",
        processes_nums=10,

        # 运存不足，无法满足 day 的运存消耗
        cycle="week",
        financial_cycle="quarter",
        start_date=start_date_,
        end_date=end_date_,
        storage_dir_name=monitor_dir,
    )
    analyzer.run()

# --------------------------------------------
def data_convert() -> None:
    """分析因子 -> 量化因子"""
    convert = DataConvert(
        source_dir=monitor_dir,
        storage_dir=monitor_dir
    )
    convert.run()


# --------------------------------------------
def get_factors_setting(
        factors_name: list[str]
) -> list:
    return [
        FACTOR_LIBRARY[factor] for factor in factors_name
    ]


# --------------------------------------------
def factor_monitor() -> None:
    """因子监控（月）"""
    monitor = FactorMonitor(
        source_dir=monitor_dir,
        storage_dir=monitor_dir,
        factors_setting=factors_setting,
        cycle="week",
        class_level="一级行业",
        lag_period=1,
        group_nums=10
    )
    monitor.stock_crawl()


# ------------------------- 执行入口 -------------------------
if __name__ == "__main__":
    # 监控因子设置
    factors = ["对数市值", "市净率倒数"]
    factors_setting = get_factors_setting(factors)

    # 路径参数
    monitor_dir = "20250530W"

    # 日期参数
    start_date = "2022-03-31"
    end_date = "2025-05-30"

    # quant_analysis(start_date, end_date)
    # data_convert()
    factor_monitor()
