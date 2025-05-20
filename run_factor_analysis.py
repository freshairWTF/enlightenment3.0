from quant.factor_service import FactorAnalyzer

from download.support_service import SupportDataUpdater

# --------------------------------------------------
def update_support():
    """更新支持数据"""
    updater = SupportDataUpdater(
        start_date="2000-01-01",
        end_date="2025-05-13",
        get_listed_code=False        # 仅更新交易日历 = False
    )
    updater.run({updater.trading_calendar: True})


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="20250502-WEEK-混合",
        index_code="000300",
        factors_name=[
            "经营净利润","净利润","归属于母公司所有者的净利润","利润总额","营业利润","营业收入","毛利",
            "息税前利润","核心利润","分配股利、利润或偿付利息所支付的现金","购买商品、接受劳务支付的现金",
            "经营活动产生的现金流量净额","投资活动现金流出小计","投资活动现金流入小计","销售商品、提供劳务收到的现金",
            "支付的其他与经营活动有关的现金","支付给职工以及为职工支付的现金","偿还债务支付的现金","筹资活动现金流出小计",
            "筹资活动现金流入小计","购建固定资产、无形资产和其他长期资产支付的现金","经营活动现金流出小计",
            "经营活动现金流入小计","取得借款收到的现金","取得投资收益所收到的现金","收到的其他与经营活动有关的现金",
            "收到的其他与投资活动有关的现金","收到的税费返还","支付的各项税费","支付的其他与投资活动有关的现金",
            "投资活动产生的现金流量净额",
            "累加收益率_0.09","累加收益率_0.17","累加收益率_0.25",
            "换手率均线_0.25","换手率均线_0.5","换手率均线_1","换手率均线_1.5",
            "换手率标准差_0.25","换手率标准差_0.5","换手率标准差_1","换手率标准差_1.5","换手率标准差_2",
            "close",
        ],
        cycle="week",
        standardization=True,
        mv_neutral=True,
        industry_neutral=True,
        restructure=False,

        group_mode="frequency",
        group_nums=10,

        # ！！！# 若大于1，则在因子端需要计算连续n日的收益率
        lag_period=1
    )
    # 运存限制，使用单进程
    analyzer.run()


if __name__ == "__main__":
    # update_support()
    factor_analysis()
