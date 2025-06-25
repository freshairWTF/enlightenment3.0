"""
财报更新：
    -1 财报
    -2 股东
    -3 股本
"""
from download.fundamental_service import Crawler, Cleaner


# --------------------------------------------------
def download_stock_fundamental():
    """下载股票基本面数据"""
    downloader = Crawler(
        data_name="financial_data",
        start_date="2000-03-31",            # 起始时间
        end_date="2025-03-31",              # 结束时间

        code="688330",                      # 代码：无需sh/sz前缀
        filter_mode="from_code",                  # 可选：None/all/from_code
        industry_info=None,                 # 指定行业

        announcement_max_page=1000          # 公告标题最大页数
    )
    downloader.stock_crawl()
    downloader.future_crawl()

# --------------------------------------------------
def download_future_fundamental():
    """下载期货基本面数据"""
    downloader = Crawler(
        data_name="cffex_ccpm",
        start_date="2010-04-16",            # 起始时间
        end_date="2025-06-25",              # 结束时间
        code="IF",                          # 代码
    )
    downloader.future_crawl()

# --------------------------------------------------
def clean_fundamental():
    """基本面数据清洗"""
    cleaner = Cleaner(
        data_name="top_ten_shareholders",
        num_processes=10,                   # 多进程核数

        code="603896",                      # 代码：无需sh/sz前缀
        filter_mode=None,                  # 可选：None/all/from_code
        industry_info=None                  # 指定行业
    )
    cleaner.run()
    # cleaner.multi_run()


# ------------------------- 执行入口 -------------------------
if __name__ == "__main__":
    # download_stock_fundamental()
    download_future_fundamental()
    # clean_fundamental()
