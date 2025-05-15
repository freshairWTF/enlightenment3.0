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
        source_dir="涨跌停次数",
        index_code="000300",
        factors_name=[
            "涨停次数_0.09", "涨停次数_0.17", "涨停次数_0.25", "涨停次数_0.5", "涨停次数_1", "涨停次数_1.5", "涨停次数_2",
            "跌停次数_0.09", "跌停次数_0.17", "跌停次数_0.25", "跌停次数_0.5", "跌停次数_1", "跌停次数_1.5", "跌停次数_2",
        ],
        cycle="week",
        standardization=True,
        mv_neutral=False,
        industry_neutral=False,
        restructure=False,

        group_mode="distant",
        group_nums=10,

        # ！！！# 若大于1，则在因子端需要计算连续n日的收益率
        lag_period=1
    )
    # 运存限制，使用单进程
    analyzer.run()


if __name__ == "__main__":
    # update_support()
    factor_analysis()
