from download.fundamental_service import Crawler, Cleaner


# --------------------------------------------------
def download_fundamental():
    """下载基本面数据"""
    downloader = Crawler(
        data_name="bonus_financing",
        start_date="2000-03-31",            # 起始时间
        end_date="2025-05-02",              # 结束时间

        code="000001",                      # 代码：无需sh/sz前缀
        filter_mode="all",                  # 可选：None/all/from_code
        industry_info=None,                 # 指定行业

        announcement_max_page=1000          # 公告标题最大页数
    )
    downloader.run()


# --------------------------------------------------
def clean_fundamental():
    """基本面数据清洗"""
    cleaner = Cleaner(
        data_name="bonus_financing",
        num_processes=10,                   # 多进程核数

        code="002852",                      # 代码：无需sh/sz前缀
        filter_mode="all",                  # 可选：None/all/from_code
        industry_info=None                  # 指定行业
    )
    cleaner.multi_run()


# ------------------------- 执行入口 -------------------------
if __name__ == "__main__":
    # download_fundamental()
    clean_fundamental()
