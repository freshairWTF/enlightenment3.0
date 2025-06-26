from factor.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="barra因子",
        index_code="000300",
        factors_name=[
            "市现率", "barra动量", "barra市场贝塔", "营业收入_yoy", "净利润_yoy"
        ],
        cycle="week",

        transfer_mode=None,
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
