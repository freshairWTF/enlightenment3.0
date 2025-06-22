from factor.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="模型因子训练集",
        index_code="000300",
        factors_name=[
            "投资活动现金流入小计", "销售商品、提供劳务收到的现金",
            "支付的其他与经营活动有关的现金", "支付给职工以及为职工支付的现金","偿还债务支付的现金",
            "筹资活动现金流出小计", "筹资活动现金流入小计", "购建固定资产、无形资产和其他长期资产支付的现金",
            "经营活动现金流出小计", "经营活动现金流入小计", "取得借款收到的现金", "取得投资收益所收到的现金",
            "收到的其他与经营活动有关的现金", "收到的其他与投资活动有关的现金", "收到的税费返还",
            "支付的各项税费", "支付的其他与投资活动有关的现金", "投资活动产生的现金流量净额"
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
