from factor.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="模型因子训练集",
        index_code="000300",
        factors_name=[
            "市现率倒数","市销率倒数","市销率倒数_rolling_normalized","市净率倒数","实际收益率",
            "核心利润盈利市值比","周期市盈率倒数"
        ],
        cycle="week",

        transfer_mode="yeo_johnson",
        standardization=True,
        mv_neutral=False,
        industry_neutral=False,

        group_mode="frequency",
        group_nums=10,
        lag_period=1
    )
    # 运存限制，使用单进程
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    factor_analysis()
