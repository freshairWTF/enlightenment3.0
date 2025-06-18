from factor.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="非线性对数市值",
        index_code="000300",
        factors_name=[
            "非线性对数市值", "非线性对数流通市值",
        ],
        cycle="week",
        standardization=True,
        mv_neutral=False,
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
