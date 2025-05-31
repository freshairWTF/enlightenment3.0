from quant.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="20250502-WEEK-混合",
        index_code="000300",
        factors_name=[
            "营业收入"
            # "换手率均线_0.25",
            # "阳线次数_0.09", "阳线次数_0.17", "阳线次数_0.09", "阳线次数_0.25", "阳线次数_0.5", "阳线次数_1",
            # "阳线次数_1.5", "阳线次数_2",
            # "阴线次数_0.09", "阴线次数_0.17", "阴线次数_0.09", "阴线次数_0.25", "阴线次数_0.5", "阴线次数_1",
            # "阴线次数_1.5", "阴线次数_2",
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
