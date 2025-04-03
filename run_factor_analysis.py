from quant.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="202503M",
        factors_name=[
            "ma_atr_0.5"
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
