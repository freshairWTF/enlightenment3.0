from factor.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="模型因子训练集",
        index_code="000300",
        factors_name=[
            "经营净利润", "净利润", "归属于母公司所有者的净利润", "利润总额", "营业利润", "营业收入",
            "毛利", "息税前利润","核心利润"
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
