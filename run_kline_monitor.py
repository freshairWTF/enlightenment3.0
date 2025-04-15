from analysis.kline_monitor import KlineMonitor
from constant.monitor_setting import Kline


# --------------------------------------------
def kline_monitor():
    """量价监控（日、周、月）"""
    monitor = KlineMonitor(
        index_code="000300",
        params=Kline(),

        start_date="2024-08-30",
        end_date="2025-04-14",
        storage_dir_name="2025-04-14",
        target_info=["002234"],
        cycle="day",
        processes_nums=10
    )
    monitor.run()


if __name__ == "__main__":
    kline_monitor()
