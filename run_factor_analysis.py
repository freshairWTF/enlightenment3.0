from quant.factor_service import FactorAnalyzer
from constant.path_config import DataPATH


# --------------------------------------------
def factor_analysis():
    """单因子分析"""
    analyzer = FactorAnalyzer(
        source_dir="202503M",
        index_path=DataPATH.INDEX_KLINE_DATA / "month" / "000300",
        factors_name=[
            "非流动资产合计"
            # "非流动资产合计", "少数股东权益", "应收票据", "应收账款", "实收资本",
            # "库存股", "存货", "所有者权益", "负债和所有者权益",
            # "利润总额", "资产减值损失", "信用减值损失", "归属于母公司所有者的净利润",
            # "净利润", "营业利润", "营业收入",
            # "投资活动现金流入小计", "投资活动现金流出小计", "收回投资所收到的现金",
            # "取得投资收益所收到的现金", "处置固定资产、无形资产和其他长期资产所收回的现金净额",
            # "收到的其他与投资活动有关的现金", "购建固定资产、无形资产和其他长期资产支付的现金",
            # "投资所支付的现金", "取得子公司及其他营业单位支付的现金净额", "支付的其他与投资活动有关的现金",
            # "投资活动产生的现金流量净额", "筹资活动现金流入小计", "筹资活动现金流出小计",
            # "筹资活动产生的现金流量净额", "吸收投资收到的现金", "取得借款收到的现金",
            # "发行债券收到的现金", "收到其他与筹资活动有关的现金", "偿还债务支付的现金",
            # "分配股利、利润或偿付利息所支付的现金", "支付其他与筹资活动有关的现金", "经营活动现金流入小计",
            # "经营活动现金流出小计", "经营活动产生的现金流量净额", "销售商品、提供劳务收到的现金",
            # "收到的税费返还", "收到的其他与经营活动有关的现金", "购买商品、接受劳务支付的现金",
            # "支付给职工以及为职工支付的现金", "支付的各项税费", "支付的其他与经营活动有关的现金",
            # "毛利率", "核心利润率", "营业净利率", "核心利润净利率", "权益净利率",
            # "归母权益净利率", "总资产净利率", "净经营资产净利率", "经营净利率",
            # "销售利润率", "经营差异率", "杠杆贡献率", "核心利润占比", "经营性资产收益率",
            # "金融性资产收益率", "毛利", "投资利润", "杂项利润", "营业外收支净额",
            # "息税前利润", "息税前利润率", "核心利润",
            # "存货周转天数", "应收票据及应收账款周转天数", "应付票据及应付账款周转天数",
            # "现金转换周期", "固定资产周转率", "总资产周转率", "单位营收所需的经营性营运资本",
            # "营业周期", "固定资产周转天数", "流动资产周转天数", "非流动资产周转天数",
            # "总资产周转天数", "净经营资产周转率", "固定资产原值推动力", "固定资产净值推动力",
            # "流动比率", "速动比率", "现金比率", "资产负债率", "有息负债率",
            # "债务资本化比率", "长期资本化比率", "利息保障倍数", "现金流量利息保障倍数",
            # "净财务杠杆", "现金流量比率",
            # "内含增长率", "可持续增长率", "利润留存率", "股利支付率", "扩张倍数",
            # "收缩倍数"
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
