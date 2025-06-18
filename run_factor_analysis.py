from factor.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="带波动率过滤的累计收益率",
        index_code="000300",
        factors_name=[
            "波动率过滤的累加收益率_0.09",
        ],
        cycle="week",
        standardization=True,
        mv_neutral=True,
        industry_neutral=True,
        restructure=False,

        group_mode="frequency",
        group_nums=10,

        lag_period=1
    )
    # 运存限制，使用单进程
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    factor_analysis()
