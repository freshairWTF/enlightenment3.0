from factor.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="模型因子训练集",
        index_code="000300",
        factors_name=[
            "换手率均线_0.5",
            "换手率均线_1", "换手率均线_1.5",
            "换手率标准差_0.25", "换手率标准差_0.5", "换手率标准差_1",
            "换手率标准差_1.5", "换手率标准差_2"
        ],
        cycle="week",

        transfer_mode="yeo_johnson",
        standardization=True,
        mv_neutral=True,
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
