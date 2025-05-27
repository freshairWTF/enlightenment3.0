from quant.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="公司治理指标",
        index_code="000300",
        factors_name=[
            "前十大股东持股比例", "大股东持股变化率", "z指数",
        ],
        cycle="week",
        standardization=True,
        mv_neutral=False,
        industry_neutral=False,
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
