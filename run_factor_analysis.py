from quant.factor_service import FactorAnalyzer


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="202503M",
        factors_name=[
            "投资活动现金流入小计_yoy", "投资活动现金流出小计_yoy", "收回投资所收到的现金_yoy",
            "取得投资收益所收到的现金_yoy", "处置固定资产、无形资产和其他长期资产所收回的现金净额_yoy",
            "收到的其他与投资活动有关的现金_yoy", "购建固定资产、无形资产和其他长期资产支付的现金_yoy",
            "投资所支付的现金_yoy", "取得子公司及其他营业单位支付的现金净额_yoy", "支付的其他与投资活动有关的现金_yoy",
            "投资活动产生的现金流量净额_yoy", "筹资活动现金流入小计_yoy", "筹资活动现金流出小计_yoy",
            "筹资活动产生的现金流量净额_yoy", "吸收投资收到的现金_yoy", "取得借款收到的现金_yoy",
            "发行债券收到的现金_yoy", "收到其他与筹资活动有关的现金_yoy", "偿还债务支付的现金_yoy",
            "分配股利、利润或偿付利息所支付的现金_yoy", "支付其他与筹资活动有关的现金_yoy", "经营活动现金流入小计_yoy",
            "经营活动现金流出小计_yoy", "经营活动产生的现金流量净额_yoy", "销售商品、提供劳务收到的现金_yoy",
            "收到的税费返还_yoy", "收到的其他与经营活动有关的现金_yoy", "购买商品、接受劳务支付的现金_yoy",
            "支付给职工以及为职工支付的现金_yoy", "支付的各项税费_yoy", "支付的其他与经营活动有关的现金_yoy"
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
