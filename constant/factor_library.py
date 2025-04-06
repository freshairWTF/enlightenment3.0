from constant.quant_setting import FactorSetting


# 整体股票因子
OVERALL_FACTOR = {
    # 时序估值
    "市销率倒数_normalized": FactorSetting(
        factor_name="市销率倒数_normalized",
        entire_filter=True,
        overall_filter=True,
        small_filter=True
    ),
    # 估值
    "周期市盈率倒数": FactorSetting(
        factor_name="周期市盈率倒数",
        entire_filter=True,
        overall_filter=True,
        large_filter=True
    ),
    # 规模
    "经营性营运资本": FactorSetting(
        factor_name="经营性营运资本",
        entire_filter=True
    ),
    # 利润
    # "毛利": FactorSetting(
    #     factor_name="毛利",
    #     entire_filter=True,
    #     overall_filter=True,
    # ),
    "经营净利润": FactorSetting(
        factor_name="经营净利润",
        entire_filter=True,
        overall_filter=True,
    ),
    # 现金流
    "经营活动产生的现金流量净额": FactorSetting(
        factor_name="经营活动产生的现金流量净额",
        entire_filter=True,
        overall_filter=True
    ),
    # 成长
    "购建固定资产、无形资产和其他长期资产支付的现金": FactorSetting(
        factor_name="购建固定资产、无形资产和其他长期资产支付的现金",
        entire_filter=True
    ),
}


FACTOR_LIBRARY = {
    # 估值
    "对数市值": FactorSetting(
        factor_name="对数市值",
        market_value_neutral=False,
        reverse=True,
        entire_filter=True,
        small_filter=True
    ),
    "对数市值_normalized": FactorSetting(
        factor_name="对数市值_normalized",
        market_value_neutral=False,
        reverse=True,
        small_filter=True
    ),
    "市值": FactorSetting(
        factor_name="市值",
        market_value_neutral=False,
        reverse=True,
        entire_filter=True,
        small_filter=True,
    ),
    "市值_normalized": FactorSetting(
        factor_name="市值_normalized",
        market_value_neutral=False,
        reverse=True,
        small_filter=True,
    ),
    "市现率倒数": FactorSetting(
        factor_name="市现率倒数",
        overall_filter=True
    ),
    "市销率倒数": FactorSetting(
        factor_name="市销率倒数",
        entire_filter=True
    ),
    "市销率倒数_normalized": FactorSetting(
        factor_name="市销率倒数_normalized",
        entire_filter=True,
        overall_filter=True,
        small_filter=True
    ),
    "市净率倒数": FactorSetting(
        factor_name="市净率倒数",
        overall_filter=True,
        entire_filter=True,
        small_filter=True
    ),
    "实际收益率": FactorSetting(
        factor_name="实际收益率",
        overall_filter=True,
    ),
    "核心利润盈利市值比": FactorSetting(
        factor_name="核心利润盈利市值比",
        entire_filter=True,
        overall_filter=True,
    ),
    "周期市盈率倒数": FactorSetting(
        factor_name="周期市盈率倒数",
        entire_filter=True,
        overall_filter=True,
        large_filter=True
    ),

    # 财务
    # 资产负债表
    "负债和所有者权益": FactorSetting(
        factor_name="负债和所有者权益",
        entire_filter=True,
        overall_filter=True
    ),
    "净经营资产": FactorSetting(
        factor_name="净经营资产",
        entire_filter=True,
        overall_filter=True
    ),
    "所有者权益": FactorSetting(
        factor_name="所有者权益",
        entire_filter=True,
        overall_filter=True
    ),
    "存货": FactorSetting(
        factor_name="存货",
        entire_filter=True
    ),
    "非流动资产合计": FactorSetting(
        factor_name="非流动资产合计",
        entire_filter=True
    ),
    "金融性负债": FactorSetting(
        factor_name="金融性负债",
        entire_filter=True
    ),
    "金融性资产": FactorSetting(
        factor_name="金融性资产",
        entire_filter=True
    ),
    "经营性流动负债": FactorSetting(
        factor_name="经营性流动负债",
        entire_filter=True
    ),
    "经营性流动资产": FactorSetting(
        factor_name="经营性流动资产",
        entire_filter=True,
        overall_filter=True
    ),
    "经营性营运资本": FactorSetting(
        factor_name="经营性营运资本",
        entire_filter=True
    ),
    "经营性长期负债": FactorSetting(
        factor_name="经营性长期负债",
        entire_filter=True
    ),
    "经营性长期资产": FactorSetting(
        factor_name="经营性长期资产",
        entire_filter=True
    ),
    "净负债": FactorSetting(
        factor_name="净负债",
        entire_filter=True
    ),
    "净经营性长期资产": FactorSetting(
        factor_name="净经营性长期资产",
        entire_filter=True
    ),
    "少数股东权益": FactorSetting(
        factor_name="少数股东权益",
        entire_filter=True
    ),
    "实收资本": FactorSetting(
        factor_name="实收资本",
        entire_filter=True
    ),

    # 利润表
    "经营净利润": FactorSetting(
        factor_name="经营净利润",
        entire_filter=True,
        overall_filter=True,
    ),
    "净利润": FactorSetting(
        factor_name="净利润",
        entire_filter=True,
        overall_filter=True,
    ),
    "归属于母公司所有者的净利润": FactorSetting(
        factor_name="归属于母公司所有者的净利润",
        entire_filter=True,
        overall_filter=True,
    ),
    "利润总额": FactorSetting(
        factor_name="利润总额",
        entire_filter=True,
        overall_filter=True,
    ),
    "营业利润": FactorSetting(
        factor_name="营业利润",
        entire_filter=True,
        overall_filter=True,
    ),
    "营业收入": FactorSetting(
        factor_name="营业收入",
        entire_filter=True,
        overall_filter=True,
    ),
    "毛利": FactorSetting(
        factor_name="毛利",
        entire_filter=True,
        overall_filter=True,
    ),
    "息税前利润": FactorSetting(
        factor_name="息税前利润",
        entire_filter=True,
        overall_filter=True,
    ),
    "核心利润": FactorSetting(
        factor_name="核心利润",
        entire_filter=True,
        overall_filter=True,
    ),

    # 现金流量表
    "分配股利、利润或偿付利息所支付的现金": FactorSetting(
        factor_name="分配股利、利润或偿付利息所支付的现金",
        entire_filter=True,
        overall_filter=True
    ),
    "购买商品、接受劳务支付的现金": FactorSetting(
        factor_name="购买商品、接受劳务支付的现金",
        entire_filter=True,
        overall_filter=True
    ),
    "经营活动产生的现金流量净额": FactorSetting(
        factor_name="经营活动产生的现金流量净额",
        entire_filter=True,
        overall_filter=True
    ),
    "投资活动现金流出小计": FactorSetting(
        factor_name="投资活动现金流出小计",
        entire_filter=True,
        overall_filter=True
    ),
    "投资活动现金流入小计": FactorSetting(
        factor_name="投资活动现金流入小计",
        entire_filter=True
    ),
    "销售商品、提供劳务收到的现金": FactorSetting(
        factor_name="销售商品、提供劳务收到的现金",
        entire_filter=True,
        overall_filter=True
    ),
    "支付的其他与经营活动有关的现金": FactorSetting(
        factor_name="支付的其他与经营活动有关的现金",
        entire_filter=True,
        overall_filter=True
    ),
    "支付给职工以及为职工支付的现金": FactorSetting(
        factor_name="支付给职工以及为职工支付的现金",
        entire_filter=True,
        overall_filter=True
    ),
    "偿还债务支付的现金": FactorSetting(
        factor_name="偿还债务支付的现金",
        entire_filter=True
    ),
    "筹资活动现金流出小计": FactorSetting(
        factor_name="筹资活动现金流出小计",
        entire_filter=True
    ),
    "筹资活动现金流入小计": FactorSetting(
        factor_name="筹资活动现金流入小计",
        entire_filter=True
    ),
    "购建固定资产、无形资产和其他长期资产支付的现金": FactorSetting(
        factor_name="购建固定资产、无形资产和其他长期资产支付的现金",
        entire_filter=True
    ),
    "经营活动现金流出小计": FactorSetting(
        factor_name="经营活动现金流出小计",
        entire_filter=True,
        overall_filter=True
    ),
    "经营活动现金流入小计": FactorSetting(
        factor_name="经营活动现金流入小计",
        entire_filter=True,
        overall_filter=True
    ),
    "取得借款收到的现金": FactorSetting(
        factor_name="取得借款收到的现金",
        entire_filter=True
    ),
    "取得投资收益所收到的现金": FactorSetting(
        factor_name="取得投资收益所收到的现金",
        entire_filter=True
    ),
    "收到的其他与经营活动有关的现金": FactorSetting(
        factor_name="收到的其他与经营活动有关的现金",
        entire_filter=True
    ),
    "收到的其他与投资活动有关的现金": FactorSetting(
        factor_name="收到的其他与投资活动有关的现金",
        entire_filter=True
    ),
    "收到的税费返还": FactorSetting(
        factor_name="收到的税费返还",
        entire_filter=True
    ),
    "支付的各项税费": FactorSetting(
        factor_name="支付的各项税费",
        entire_filter=True,
        overall_filter=True
    ),
    "支付的其他与投资活动有关的现金": FactorSetting(
        factor_name="支付的其他与投资活动有关的现金",
        entire_filter=True
    ),
    "投资活动产生的现金流量净额": FactorSetting(
        factor_name="投资活动产生的现金流量净额",
        reverse=True,
        entire_filter=True
    ),

    # 比率

    # 量价因子
    "累加收益率": FactorSetting(
        factor_name="累加收益率_0.25",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
        small_filter=True
    ),
    "累加收益率": FactorSetting(
        factor_name="累加收益率_0.25",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
        small_filter=True
    ),
}
