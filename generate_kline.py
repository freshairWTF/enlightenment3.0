from download.candlestick_service import Downloader, Cleaner
from constant.path_config import DataPATH

# --------------------------------------------------
def download_kline():
    """k线下载"""
    downloader = Downloader(
        dir_path=DataPATH.KLINE_DATA,
        download_object="index",            # 可选：stock/index
        category="day",                     # 指数仅支持 day
        adjust_flag="2",                    # 复权模式
        start_date="2025-05-26",            # 起始时间
        end_date="2025-05-30",              # 结束时间
        code="sz.301626",                   # 代码：需要sh/sz前缀
        filter_mode="all",                  # 可选：None/all/from_code
        industry_info=None,                 # 指定行业
        pause_time=0.3
    )
    downloader.run()

# --------------------------------------------------
def clean_kline():
    """k线清洗"""
    cleaner = Cleaner(
        clean_object="index",                # 可选：stock/index
        num_processes=10,                    # 多进程核数
        adjust_mode="split_adjusted",        # 复权模式
        code="688755",                       # 代码：无需sh/sz前缀
        filter_mode="all",                   # 可选：None/all/from_code
        industry_info=None                   # 指定行业
    )
    cleaner.run()


# ------------------------- 执行入口 -------------------------
if __name__ == "__main__":
    download_kline()
    clean_kline()
