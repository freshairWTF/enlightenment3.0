from quant.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="202503M",
        factors_name=[
            # "换手率标准差_0.5", "换手率标准差_1", "换手率标准差_1.5", "换手率标准差_2",
            "累加收益率_0.17", "累加收益率_0.09"
            # "斜率_0.25", "斜率_0.5", "斜率_1", "斜率_1.5", "斜率_2",
        ],
        cycle="month",
        standardization=True,
        mv_neutral=True,
        industry_neutral=True,
        restructure=False,
        group_nums=10
    )
    # 运存限制，使用单进程
    analyzer.run()


if __name__ == "__main__":
    factor_analysis()
