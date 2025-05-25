from quant.factor_service import FactorAnalyzer

from download.support_service import SupportDataUpdater

# --------------------------------------------------
def update_support():
    """更新支持数据"""
    updater = SupportDataUpdater(
        start_date="2000-01-01",
        end_date="2025-05-13",
        get_listed_code=False        # 仅更新交易日历 = False
    )
    updater.run({updater.trading_calendar: True})


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="20250502-WEEK-混合",
        index_code="000300",
        factors_name=[
            # "对数市值","累加收益率_0.17","累加收益率_0.25",
            # "换手率均线_0.25","换手率均线_0.5","换手率均线_1","换手率均线_1.5",
            # "换手率标准差_0.25","换手率标准差_0.5","换手率标准差_1","换手率标准差_1.5","换手率标准差_2",
            "close",
        ],
        cycle="week",
        standardization=True,
        mv_neutral=False,
        industry_neutral=False,
        restructure=False,

        group_mode="frequency",
        group_nums=10,

        # ！！！# 若大于1，则在因子端需要计算连续n日的收益率
        lag_period=1
    )
    # 运存限制，使用单进程
    analyzer.run()


if __name__ == "__main__":
    # update_support()
    factor_analysis()
