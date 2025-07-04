"""
财报更新：
    -1 财报
    -2 股东
    -3 股本
"""
from download.fundamental_service import Crawler, Cleaner


# --------------------------------------------------
def download_fundamental():
    """下载基本面数据"""
    downloader = Crawler(
        data_name="top_ten_circulating_shareholders",
        start_date="2000-03-31",            # 起始时间
        end_date="2025-03-31",              # 结束时间

        code="688330",                      # 代码：无需sh/sz前缀
        filter_mode="from_code",                  # 可选：None/all/from_code
        industry_info=None,                 # 指定行业

        announcement_max_page=1000          # 公告标题最大页数
    )
    downloader.run()

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
    download_fundamental()
    # clean_fundamental()
