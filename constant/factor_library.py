"""因子库"""
from constant.quant_setting import FactorSetting

"""
方向会变的因子，不能指定方向

短线、中线、长线 因子根据半衰期来定

对数市值 不需要市值中性？

不需要写那么多个字典
需要什么，就加条件，比如 半衰期、过滤模式、适应市场、因子类型
"""

# ----------------------------------------------------------
ALPHA_FACTOR_LIBRARY = {
    # 估值因子
    "对数市值-1": FactorSetting(
        factor_name="对数市值",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        reverse=True,
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),
    "对数市值-2": FactorSetting(
        factor_name="对数市值",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_small_cap_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),

    "对数市值_rolling_normalized-1": FactorSetting(
        factor_name="对数市值_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "对数市值_rolling_normalized-2": FactorSetting(
        factor_name="对数市值_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        half_life=2,
        reverse=True,
         filter_mode = "_overall_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "对数市值_rolling_normalized-3": FactorSetting(
        factor_name="对数市值_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        half_life=2,
        reverse=True,
         filter_mode = "_small_cap_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),

    "市值-1": FactorSetting(
        factor_name="市值",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_small_cap_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),

    "市值_rolling_normalized-1": FactorSetting(
        factor_name="市值_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        half_life=2,
        reverse=True,
         filter_mode = "_small_cap_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),

    "市销率倒数_rolling_normalized-1": FactorSetting(
        factor_name="市销率倒数_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        half_life=3,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
    ),
    "市销率倒数_rolling_normalized-2": FactorSetting(
        factor_name="市销率倒数_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        half_life=4,
         filter_mode = "_overall_filter",
        bull_market=True,
        bear_market=True,
    ),
    "市销率倒数_rolling_normalized-3": FactorSetting(
        factor_name="市销率倒数_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        cycle="week",
        half_life=2,
         filter_mode = "_small_cap_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),

    # 规模因子
    "实收资本-1": FactorSetting(
        factor_name="实收资本",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        cycle="week",
        half_life=4,
         filter_mode = "_small_cap_filter",
        bull_market=True,
    ),
    "所有者权益-1": FactorSetting(
        factor_name="所有者权益",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),

    # 质量因子
    "归属于母公司所有者的净利润-1": FactorSetting(
        factor_name="归属于母公司所有者的净利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),
    "核心利润-1": FactorSetting(
        factor_name="核心利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "经营净利润-1": FactorSetting(
        factor_name="经营净利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "净利润-1": FactorSetting(
        factor_name="净利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),
    "利润总额-1": FactorSetting(
        factor_name="净利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),
    "毛利-1": FactorSetting(
        factor_name="毛利",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "息税前利润-1": FactorSetting(
        factor_name="息税前利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),
    "营业利润-1": FactorSetting(
        factor_name="营业利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),

    # 现金流量因子
    "经营活动产生的现金流量净额-1": FactorSetting(
        factor_name="经营活动产生的现金流量净额",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),
    "经营活动现金流入小计-1": FactorSetting(
        factor_name="经营活动现金流入小计",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),
    "销售商品、提供劳务收到的现金-1": FactorSetting(
        factor_name="销售商品、提供劳务收到的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),
    "支付的各项税费-1": FactorSetting(
        factor_name="支付的各项税费",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),
    "支付给职工以及为职工支付的现金-1": FactorSetting(
        factor_name="支付给职工以及为职工支付的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        cycle="week",
         filter_mode = "_entire_filter",
        bear_market=True,
        shocking_market=True
    ),

    # 流动性因子
    "换手率均线-0.25-1": FactorSetting(
        factor_name="换手率均线_0.25",
        primary_classification="技术面因子",
        secondary_classification="流动性因子",
        cycle="week",
        half_life=8,
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "换手率均线-0.25-2": FactorSetting(
        factor_name="换手率均线_0.25",
        primary_classification="技术面因子",
        secondary_classification="流动性因子",
        cycle="week",
        half_life=10,
        reverse=True,
         filter_mode = "_overall_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "换手率均线-0.5-1": FactorSetting(
        factor_name="换手率均线_0.5",
        primary_classification="技术面因子",
        secondary_classification="流动性因子",
        cycle="week",
        half_life=12,
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
    ),
    "换手率均线-0.5-2": FactorSetting(
        factor_name="换手率均线_0.5",
        primary_classification="技术面因子",
        secondary_classification="流动性因子",
        cycle="week",
        reverse=True,
         filter_mode = "_overall_filter",
        bear_market=True,
        shocking_market=True
    ),
    "换手率均线-0.5-3": FactorSetting(
        factor_name="换手率均线_0.5",
        primary_classification="技术面因子",
        secondary_classification="流动性因子",
        cycle="week",
        reverse=True,
         filter_mode = "_mega_cap_filter",
        bear_market=True,
        shocking_market=True
    ),

    # 流动性风险因子
    "换手率标准差-0.25-1": FactorSetting(
        factor_name="换手率标准差_0.25",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        half_life=5,
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "换手率标准差-0.25-2": FactorSetting(
        factor_name="换手率标准差_0.25",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        half_life=5,
        reverse=True,
         filter_mode = "_overall_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "换手率标准差-0.25-3": FactorSetting(
        factor_name="换手率标准差_0.25",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
         filter_mode = "_large_cap_filter",
        shocking_market=True
    ),
    "换手率标准差-0.25-4": FactorSetting(
        factor_name="换手率标准差_0.25",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
         filter_mode = "_mega_cap_filter",
        bear_market=True,
        shocking_market=True
    ),
    "换手率标准差-0.5-1": FactorSetting(
        factor_name="换手率标准差_0.5",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        half_life=10,
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "换手率标准差-0.5-2": FactorSetting(
        factor_name="换手率标准差_0.5",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        half_life=11,
        reverse=True,
         filter_mode = "_overall_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "换手率标准差-0.5-3": FactorSetting(
        factor_name="换手率标准差_0.5",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
         filter_mode = "_large_cap_filter",
        shocking_market=True
    ),
    "换手率标准差-1-1": FactorSetting(
        factor_name="换手率标准差_1",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
    ),
    "换手率标准差-1-2": FactorSetting(
        factor_name="换手率标准差_1",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        reverse=True,
         filter_mode = "_overall_filter",
        bull_market=True,
        bear_market=True,
    ),
    "换手率标准差-1-3": FactorSetting(
        factor_name="换手率标准差_1",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
         filter_mode = "_large_cap_filter",
        shocking_market=True,
    ),
    "换手率标准差-1.5-1": FactorSetting(
        factor_name="换手率标准差_1.5",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
         filter_mode = "_large_cap_filter",
        shocking_market=True,
    ),
    "换手率标准差-1.5-2": FactorSetting(
        factor_name="换手率标准差_1.5",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
         filter_mode = "_mega_cap_filter",
        bear_market=True,
        shocking_market=True,
    ),
    "换手率标准差-2-1": FactorSetting(
        factor_name="换手率标准差_2",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
         filter_mode = "_large_cap_filter",
        shocking_market=True,
    ),

    # 动量因子
    "累加收益率-0.09-1": FactorSetting(
        factor_name="累加收益率_0.09",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "累加收益率-0.09-2": FactorSetting(
        factor_name="累加收益率_0.09",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_overall_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "累加收益率-0.09-3": FactorSetting(
        factor_name="累加收益率_0.09",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_large_cap_filter",
        bull_market=True,
        bear_market=True,
    ),
    "累加收益率-0.09-4": FactorSetting(
        factor_name="累加收益率_0.09",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_small_cap_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "累加收益率-0.17-1": FactorSetting(
        factor_name="累加收益率_0.17",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "累加收益率-0.17-2": FactorSetting(
        factor_name="累加收益率_0.17",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_overall_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "累加收益率-0.17-3": FactorSetting(
        factor_name="累加收益率_0.17",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_small_cap_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "累加收益率-0.25-1": FactorSetting(
        factor_name="累加收益率_0.25",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "累加收益率-0.25-2": FactorSetting(
        factor_name="累加收益率_0.25",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_overall_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "累加收益率-0.25-3": FactorSetting(
        factor_name="累加收益率_0.25",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        cycle="week",
        half_life=1,
        reverse=True,
         filter_mode = "_small_cap_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),

    # 行为金融因子
    "close-1": FactorSetting(
        factor_name="close",
        primary_classification="行为金融因子",
        secondary_classification="行为金融因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
         filter_mode = "_entire_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),
    "close-2": FactorSetting(
        factor_name="close",
        primary_classification="行为金融因子",
        secondary_classification="行为金融因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
         filter_mode = "_overall_filter",
        bear_market=True,
        shocking_market=True
    ),
    "close-3": FactorSetting(
        factor_name="close",
        primary_classification="行为金融因子",
        secondary_classification="行为金融因子",
        cycle="week",
        market_value_neutral=False,
        industry_neutral=False,
        reverse=True,
        filter_mode = "_small_cap_filter",
        bull_market=True,
        bear_market=True,
        shocking_market=True
    ),

    # 波动率
    "真实波幅均线_1": FactorSetting(
        factor_name="真实波幅均线_0.09",
        primary_classification="技术面因子",
        secondary_classification="波动率因子",
        reverse=True,
        filter_mode="_entire_filter",
    ),
    "真实波幅均线_2": FactorSetting(
        factor_name="真实波幅均线_0.5",
        primary_classification="技术面因子",
        secondary_classification="波动率因子",
        reverse=True,
        filter_mode="_entire_filter",
    ),
    "真实波幅均线_3": FactorSetting(
        factor_name="真实波幅均线_1",
        primary_classification="技术面因子",
        secondary_classification="波动率因子",
        reverse=True,
        filter_mode="_entire_filter",
    ),
    "真实波幅均线_4": FactorSetting(
        factor_name="真实波幅均线_1.5",
        primary_classification="技术面因子",
        secondary_classification="波动率因子",
        reverse=True,
        filter_mode="_entire_filter",
    ),
    "真实波幅均线_5": FactorSetting(
        factor_name="真实波幅均线_2",
        primary_classification="技术面因子",
        secondary_classification="波动率因子",
        reverse=True,
        filter_mode="_entire_filter",
    ),

    # 公司治理
    # "前十大股东持股变化率-1": FactorSetting(
    #     factor_name="前十大股东持股变化率",
    #     primary_classification="基本面因子",
    #     secondary_classification="公司治理因子",
    #     cycle="week",
    #     half_life=10,
    #     market_value_neutral=False,
    #     industry_neutral=False,
    #     filter_mode="_entire_filter",
    #     # bull_market=True,
    #     # bear_market=True,
    #     # shocking_market=True
    # ),
}


# ----------------------------------------------------------
RISK_FACTOR_LIBRARY = {
    "对数市值": FactorSetting(
        factor_name="对数市值",
        cycle="week",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        market_value_neutral=False,
        reverse=True,
        entire_filter=True,
        small_filter=True
    ),
}


# ----------------------------------------------------------
FACTOR_LIBRARY = {
    # 估值
    "对数市值": FactorSetting(
        factor_name="对数市值",
        cycle="week",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        market_value_neutral=False,
        reverse=True,
        entire_filter=True,
        small_filter=True
    ),
    "对数市值_rolling_normalized": FactorSetting(
        factor_name="对数市值_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        market_value_neutral=False,
        reverse=True,
        small_filter=True
    ),
    "市值": FactorSetting(
        factor_name="市值",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        market_value_neutral=False,
        reverse=True,
        entire_filter=True,
        small_filter=True,
    ),
    "市值_rolling_normalized": FactorSetting(
        factor_name="市值_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        market_value_neutral=False,
        reverse=True,
        small_filter=True,
    ),
    "市现率倒数": FactorSetting(
        factor_name="市现率倒数",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        overall_filter=True
    ),
    "市销率倒数": FactorSetting(
        factor_name="市销率倒数",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        entire_filter=True
    ),
    "市销率倒数_rolling_normalized": FactorSetting(
        factor_name="市销率倒数_rolling_normalized",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        entire_filter=True,
        overall_filter=True,
        small_filter=True
    ),
    "市净率倒数": FactorSetting(
        factor_name="市净率倒数",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        overall_filter=True,
        entire_filter=True,
        small_filter=True
    ),
    "实际收益率": FactorSetting(
        factor_name="实际收益率",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        overall_filter=True,
    ),
    "核心利润盈利市值比": FactorSetting(
        factor_name="核心利润盈利市值比",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        entire_filter=True,
        overall_filter=True,
    ),
    "周期市盈率倒数": FactorSetting(
        factor_name="周期市盈率倒数",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        entire_filter=True,
        overall_filter=True,
        large_filter=True
    ),

    # 财务
    # 资产负债表
    "负债和所有者权益": FactorSetting(
        factor_name="负债和所有者权益",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True,
        overall_filter=True
    ),
    "净经营资产": FactorSetting(
        factor_name="净经营资产",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True,
        overall_filter=True
    ),
    "所有者权益": FactorSetting(
        factor_name="所有者权益",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True,
        overall_filter=True
    ),
    "存货": FactorSetting(
        factor_name="存货",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "非流动资产合计": FactorSetting(
        factor_name="非流动资产合计",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "金融性负债": FactorSetting(
        factor_name="金融性负债",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "金融性资产": FactorSetting(
        factor_name="金融性资产",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "经营性流动负债": FactorSetting(
        factor_name="经营性流动负债",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "经营性流动资产": FactorSetting(
        factor_name="经营性流动资产",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True,
        overall_filter=True
    ),
    "经营性营运资本": FactorSetting(
        factor_name="经营性营运资本",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "经营性长期负债": FactorSetting(
        factor_name="经营性长期负债",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "经营性长期资产": FactorSetting(
        factor_name="经营性长期资产",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "净负债": FactorSetting(
        factor_name="净负债",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "净经营性长期资产": FactorSetting(
        factor_name="净经营性长期资产",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "少数股东权益": FactorSetting(
        factor_name="少数股东权益",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),
    "实收资本": FactorSetting(
        factor_name="实收资本",
        primary_classification="基本面因子",
        secondary_classification="规模因子",
        entire_filter=True
    ),

    # 利润表
    "经营净利润": FactorSetting(
        factor_name="经营净利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        entire_filter=True,
        overall_filter=True,
    ),
    "净利润": FactorSetting(
        factor_name="净利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        entire_filter=True,
        overall_filter=True,
    ),
    "归属于母公司所有者的净利润": FactorSetting(
        factor_name="归属于母公司所有者的净利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        entire_filter=True,
        overall_filter=True,
    ),
    "利润总额": FactorSetting(
        factor_name="利润总额",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        entire_filter=True,
        overall_filter=True,
    ),
    "营业利润": FactorSetting(
        factor_name="营业利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        entire_filter=True,
        overall_filter=True,
    ),
    "营业收入": FactorSetting(
        factor_name="营业收入",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        entire_filter=True,
        overall_filter=True,
    ),
    "毛利": FactorSetting(
        factor_name="毛利",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        entire_filter=True,
        overall_filter=True,
    ),
    "息税前利润": FactorSetting(
        factor_name="息税前利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        entire_filter=True,
        overall_filter=True,
    ),
    "核心利润": FactorSetting(
        factor_name="核心利润",
        primary_classification="基本面因子",
        secondary_classification="质量因子",
        entire_filter=True,
        overall_filter=True,
    ),

    # 现金流量表
    "分配股利、利润或偿付利息所支付的现金": FactorSetting(
        factor_name="分配股利、利润或偿付利息所支付的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "购买商品、接受劳务支付的现金": FactorSetting(
        factor_name="购买商品、接受劳务支付的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "经营活动产生的现金流量净额": FactorSetting(
        factor_name="经营活动产生的现金流量净额",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "投资活动现金流出小计": FactorSetting(
        factor_name="投资活动现金流出小计",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "投资活动现金流入小计": FactorSetting(
        factor_name="投资活动现金流入小计",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "销售商品、提供劳务收到的现金": FactorSetting(
        factor_name="销售商品、提供劳务收到的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "支付的其他与经营活动有关的现金": FactorSetting(
        factor_name="支付的其他与经营活动有关的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "支付给职工以及为职工支付的现金": FactorSetting(
        factor_name="支付给职工以及为职工支付的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "偿还债务支付的现金": FactorSetting(
        factor_name="偿还债务支付的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "筹资活动现金流出小计": FactorSetting(
        factor_name="筹资活动现金流出小计",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "筹资活动现金流入小计": FactorSetting(
        factor_name="筹资活动现金流入小计",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "购建固定资产、无形资产和其他长期资产支付的现金": FactorSetting(
        factor_name="购建固定资产、无形资产和其他长期资产支付的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "经营活动现金流出小计": FactorSetting(
        factor_name="经营活动现金流出小计",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "经营活动现金流入小计": FactorSetting(
        factor_name="经营活动现金流入小计",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "取得借款收到的现金": FactorSetting(
        factor_name="取得借款收到的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "取得投资收益所收到的现金": FactorSetting(
        factor_name="取得投资收益所收到的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "收到的其他与经营活动有关的现金": FactorSetting(
        factor_name="收到的其他与经营活动有关的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "收到的其他与投资活动有关的现金": FactorSetting(
        factor_name="收到的其他与投资活动有关的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "收到的税费返还": FactorSetting(
        factor_name="收到的税费返还",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "支付的各项税费": FactorSetting(
        factor_name="支付的各项税费",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True,
        overall_filter=True
    ),
    "支付的其他与投资活动有关的现金": FactorSetting(
        factor_name="支付的其他与投资活动有关的现金",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        entire_filter=True
    ),
    "投资活动产生的现金流量净额": FactorSetting(
        factor_name="投资活动产生的现金流量净额",
        primary_classification="基本面因子",
        secondary_classification="现金流量因子",
        reverse=True,
        entire_filter=True
    ),

    # 比率

    # 量价因子
    "累加收益率_0.09": FactorSetting(
        factor_name="累加收益率_0.09",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
        small_filter=True
    ),
    "累加收益率_0.17": FactorSetting(
        factor_name="累加收益率_0.17",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
        small_filter=True
    ),
    "累加收益率_0.25": FactorSetting(
        factor_name="累加收益率_0.25",
        primary_classification="技术面因子",
        secondary_classification="动量因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
        small_filter=True
    ),

    "换手率均线_0.25": FactorSetting(
        factor_name="换手率均线_0.25",
        primary_classification="技术面因子",
        secondary_classification="流动性因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
    ),
    "换手率均线_0.5": FactorSetting(
        factor_name="换手率均线_0.5",
        primary_classification="技术面因子",
        secondary_classification="流动性因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
    ),
    "换手率均线_1": FactorSetting(
        factor_name="换手率均线_1",
        primary_classification="技术面因子",
        secondary_classification="流动性因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
    ),
    "换手率均线_1.5": FactorSetting(
        factor_name="换手率均线_1.5",
        primary_classification="技术面因子",
        secondary_classification="流动性因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
    ),

    "换手率标准差_1": FactorSetting(
        factor_name="换手率标准差_0.25",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
    ),
    "换手率标准差_2": FactorSetting(
        factor_name="换手率标准差_0.5",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
    ),
    "换手率标准差_3": FactorSetting(
        factor_name="换手率标准差_1",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
    ),
    "换手率标准差_4": FactorSetting(
        factor_name="换手率标准差_1.5",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
    ),
    "换手率标准差_5": FactorSetting(
        factor_name="换手率标准差_2",
        primary_classification="技术面因子",
        secondary_classification="流动性风险因子",
        reverse=True,
        entire_filter=True,
        overall_filter=True,
    ),

    # 波动率
    # "真实波幅均线_1": FactorSetting(
    #     factor_name="真实波幅均线_0.09",
    #     primary_classification="技术面因子",
    #     secondary_classification="波动率因子",
    #     reverse=True,
    #     entire_filter=True,
    #     overall_filter=True,
    # ),
    # "真实波幅均线_2": FactorSetting(
    #     factor_name="真实波幅均线_0.5",
    #     primary_classification="技术面因子",
    #     secondary_classification="波动率因子",
    #     reverse=True,
    #     entire_filter=True,
    #     overall_filter=True,
    # ),
    # "真实波幅均线_3": FactorSetting(
    #     factor_name="真实波幅均线_1",
    #     primary_classification="技术面因子",
    #     secondary_classification="波动率因子",
    #     reverse=True,
    #     entire_filter=True,
    #     overall_filter=True,
    # ),
    # "真实波幅均线_4": FactorSetting(
    #     factor_name="真实波幅均线_1.5",
    #     primary_classification="技术面因子",
    #     secondary_classification="波动率因子",
    #     reverse=True,
    #     entire_filter=True,
    #     overall_filter=True,
    # ),
    # "真实波幅均线_5": FactorSetting(
    #     factor_name="真实波幅均线_2",
    #     primary_classification="技术面因子",
    #     secondary_classification="波动率因子",
    #     reverse=True,
    #     entire_filter=True,
    #     overall_filter=True,
    # ),

    # 行为金融
    "close": FactorSetting(
        factor_name="close",
        primary_classification="行为金融因子",
        secondary_classification="行为金融因子",
        reverse=True,
        market_value_neutral=False,
        industry_neutral=False,
        entire_filter=True,
        overall_filter=True,
        small_filter=True
    ),
}


# ----------------------------------------------------------
FACTOR_TEST = {
    "对数市值": FactorSetting(
        factor_name="对数市值",
        cycle="week",
        primary_classification="基本面因子",
        secondary_classification="估值因子",
        market_value_neutral=False,
        reverse=True
    ),
    # "dastd": FactorSetting(
    #     factor_name="dastd",
    #     cycle="week",
    #     primary_classification="基本面因子",
    #     secondary_classification="估值因子",
    #     market_value_neutral=True,
    #     reverse=True
    # ),
    # "close-1": FactorSetting(
    #     factor_name="close",
    #     primary_classification="基本面因子",
    #     secondary_classification="估值因子",
    #     cycle="week",
    #     market_value_neutral=False,
    #     industry_neutral=False,
    #     reverse=True
    # ),
}
