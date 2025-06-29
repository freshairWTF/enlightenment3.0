from factor.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="barra因子",
        index_code="000300",
        factors_name=[
            # "dastd", "资产负债率", "barra换手率", "barra市场贝塔", "barra动量", "营业收入_yoy",
            # "收益率标准差_0.09", "账面市值比"
            "非线性对数市值", "对数市值"
        ],
        cycle="week",

        transfer_mode=None,
        standardization=True,
        mv_neutral=False,
        industry_neutral=True,

        group_mode="frequency",
        group_nums=10,
        lag_period=1
    )
    # 运存限制，使用单进程
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    factor_analysis()
