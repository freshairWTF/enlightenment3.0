from download.candlestick_service import Downloader, Cleaner
from constant.path_config import DataPATH


# --------------------------------------------------
def download_kline():
    """k线下载"""
    downloader = Downloader(
        dir_path=DataPATH.KLINE_DATA,
        download_object="stock",            # 可选：stock/index
        category="day",                     # 指数仅支持 day
        adjust_flag="1",                    # 复权模式
        start_date="2025-04-15",            # 起始时间
        end_date="2025-04-15",              # 结束时间
        code="sz.301665",                   # 代码：需要sh/sz前缀
        filter_mode="all",                  # 可选：None/all/from_code
        industry_info=None,                 # 指定行业
        pause_time=0.3
    )
    downloader.run()

# --------------------------------------------------
def clean_kline():
    """k线清洗"""
    cleaner = Cleaner(
        clean_object="stock",                # 可选：stock/index
        num_processes=10,                    # 多进程核数
        adjust_mode="backward_adjusted",     # 复权模式
        code="301665",                       # 代码：无需sh/sz前缀
        filter_mode=None,                   # 可选：None/all/from_code
        industry_info=None                   # 指定行业
    )
    cleaner.run()


# ------------------------- 执行入口 -------------------------
if __name__ == "__main__":
    download_kline()
    # clean_kline()
