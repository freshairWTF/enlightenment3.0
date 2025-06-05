from quant.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="20250530-WEEK-跟踪",
        index_code="000300",
        factors_name=[
            "负债和所有者权益"
            # "收益率标准差_0.09", "收益率标准差_0.17", "收益率标准差_0.09", "收益率标准差_0.25", "收益率标准差_0.5", "收益率标准差_1",
            # "收益率标准差_1.5", "收益率标准差_2",
            # "真实波幅均线_0.09", "真实波幅均线_0.17", "真实波幅均线_0.09", "真实波幅均线_0.25", "真实波幅均线_0.5", "真实波幅均线_1",
            # "真实波幅均线_1.5", "真实波幅均线_2",
        ],
        cycle="week",
        standardization=True,
        mv_neutral=True,
        industry_neutral=True,
        restructure=False,

        group_mode="frequency",
        group_nums=10,

        # ！！！# 若大于1，则在因子端需要计算连续n日的收益率
        lag_period=1
    )
    # 运存限制，使用单进程
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    factor_analysis()
