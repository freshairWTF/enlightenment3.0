from download.candlestick_service import Downloader, Cleaner
from constant.path_config import DataPATH


# --------------------------------------------------
def download_kline(flag):
    """k线下载"""
    downloader = Downloader(
        dir_path=DataPATH.KLINE_DATA,
        download_object="stock",            # 可选：stock/index
        category="day",                     # 指数仅支持 day
        adjust_flag=flag,                   # 复权模式
        start_date=start_date,              # 起始时间
        end_date=end_date,                  # 结束时间
        code="sz.301195",                   # 代码：需要sh/sz前缀
        filter_mode="all",                  # 可选：None/all/from_code
        industry_info=None,                 # 指定行业
        pause_time=0.3
    )
    downloader.run()

# --------------------------------------------------
def clean_kline(adjust_mode):
    """k线清洗"""
    cleaner = Cleaner(
        clean_object="stock",                # 可选：stock/index
        num_processes=10,                    # 多进程核数
        adjust_mode=adjust_mode,             # 复权模式
        code="000576",                       # 代码：无需sh/sz前缀
        filter_mode="all",                   # 可选：None/all/from_code
        industry_info=None                   # 指定行业
    )
    cleaner.run()


# --------------------------------------------------
def daily_update_kline():
    # ---------------------------------------
    # 未复权数据 -> 前复权数据 -> 后复权数据
    # 原始价格 -> 复权因子 | 前复权价格 -> 涨跌停数据
    # ---------------------------------------
    for adjust_flag in ["1", "2", "3"]:
        download_kline(adjust_flag)
    for adjust_mode_ in ["unadjusted", "split_adjusted", "backward_adjusted"]:
        clean_kline(adjust_mode_)


# ------------------------- 执行入口 -------------------------
if __name__ == "__main__":
    start_date = "2025-06-02"
    end_date = "2025-06-06"
    daily_update_kline()
