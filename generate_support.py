from download.support_service import SupportDataUpdater
from data_convert import DataConvert


# --------------------------------------------------
def update_support():
    """更新支持数据"""
    updater = SupportDataUpdater(
        start_date="2000-01-01",
        end_date="2100-01-01",
        get_listed_code=False        # 仅更新交易日历 = False
    )
    tasks = {
        updater.listed_nums: False,
        updater.trading_calendar: True,
        updater.industry_classification: False
    }
    updater.run(tasks)


# --------------------------------------------
def data_convert():
    """分析因子 -> 量化因子"""
    convert = DataConvert(
        source_dir="公司治理指标",
        storage_dir="公司治理指标"
    )
    convert.run()


# ------------------------- 执行入口 -------------------------
if __name__ == "__main__":
    update_support()
    # data_convert()
