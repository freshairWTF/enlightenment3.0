# encoding:utf-8
"""
数据分析
"""

import warnings
from collections import Counter

import pandas as pd

from analyze_parameters import INDIVIDUAL_STOCK_HK
from download import PARAMETERS_FOR_FINANCIAL_INDEX_OF_EASTMONEY_HK, SENIOR_FINANCIAL_INDEX_HK, PARAMETERS_FOR_VALUATION
from utils import DataLoader, DataStorage
from indicators_financial import FinancialIndicatorsHK
from statistics import IndividualStatistics
from drawer import DrawToIndividualAnalysis

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


analysis_info_dict = {
    'data_from_eastmoney_hk': {
        'financial_dir_path': 'cleaned_data/hk_financial_data_from_eastmoney',
        'data_source': 'eastMoney',
        'financial_index': FinancialIndicatorsHK,
    },
}


if __name__ == '__main__':
    # ----------------------------------------- 初始参数 ----------------------------------------- #
    # ------------ 路径类参数 ------------ #
    kline_root_path = 'kline_data'                                          # K线数据
    result_root_path = 'result_data/个股分析'                               # 分析结果

    # ------------ 一般类参数 ------------ #
    analysis_info = analysis_info_dict['data_from_eastmoney_hk']
    # 分析对象
    code = '01579'
    # 财务指标时间区间
    start_date, end_date = '2015-12-31', '2023-12-31'
    # 分析周期：'quarter'(季度)/year(年)/half(半年)
    analysis_cycle = 'year'
    # 财务科目/估值方法/统计方法
    financial_indicators = INDIVIDUAL_STOCK_HK['财务']
    value_indicators = INDIVIDUAL_STOCK_HK['估值']
    stat_indicators = INDIVIDUAL_STOCK_HK['统计']

    print('-----------------------参数初始化-----------------------')
    count = Counter(financial_indicators)
    if [num for num in count.values() if num > 1]:
        exit(f'财务指标参数有重复项')

    # 财务科目名
    financial_account_words = []
    # k线数据
    kline_df = pd.DataFrame()
    # 财务数据
    financial_df = pd.DataFrame()
    # 统计数据
    stat_df = pd.DataFrame()
    # 估值数据
    value_df = pd.DataFrame()

    # 存储路径
    storage_file_path = f'{result_root_path}/{code}-{analysis_cycle}-{start_date}-{end_date}'

    # 数据加载对象
    loader = DataLoader
    # 统计对象
    stat = IndividualStatistics
    # 存储实例
    storage = DataStorage
    # 绘图实例
    draw = DrawToIndividualAnalysis(storage_file_path, ['管理用财务报表', '指标'])

    # 能用的财务指标名字典
    parameters_for_financial_index = PARAMETERS_FOR_FINANCIAL_INDEX_OF_EASTMONEY_HK
    parameters_for_senior_financial_index = SENIOR_FINANCIAL_INDEX_HK
    # 能用的估值指标名字典
    parameters_for_valuation = PARAMETERS_FOR_VALUATION

    # 获取所需全部需要计算的科目
    calculate_financial_index = [d for d in financial_indicators if d in parameters_for_financial_index.keys()
                                 or d in parameters_for_senior_financial_index.keys()]
    # 整合未加工与预加工字典
    financial_index_comparison_dict = {**parameters_for_financial_index, **parameters_for_senior_financial_index}

    # 获取财务科目、计算指标、以及估值所需全部的科目名
    # 财务
    for financial_index in financial_indicators:
        # 需计算、未加工
        if financial_index in parameters_for_financial_index.keys():
            financial_account_words.extend(parameters_for_financial_index[financial_index])
        # 无需计算
        elif financial_index not in calculate_financial_index:
            financial_account_words.append(financial_index)
        # 需计算，预加工
        elif financial_index in parameters_for_senior_financial_index.keys():
            for factor in parameters_for_senior_financial_index[financial_index]:
                if factor in parameters_for_financial_index.keys():
                    financial_account_words.extend(parameters_for_financial_index[factor])
                else:
                    financial_account_words.append(factor)
    # 估值
    for key_word in value_indicators:
        financial_account_words.extend(parameters_for_valuation[key_word])
    # 去重
    financial_account_words = list(set(financial_account_words))

    print(f'-----------------------数据加载、处理中-----------------------')
    print(f'———— 加载财务数据 ————')
    # 财务报表数据
    path = analysis_info['financial_dir_path'] + f'/{code}.xlsx'
    financial_df = loader.get_financial_data(path, start_date, end_date, analysis_cycle,
                                             [], key_filter=False)

    print(f'———— 计算财务指标 ————')
    # 计算财务指标/合并数据
    financial = FinancialIndicatorsHK(financial_df, None, calculate_financial_index, analysis_info['data_source'])
    financial.calculate()
    financial_df = financial_df.round(2)

    print('———— 统计数据 ————')
    stat = stat(financial_df, kline_df, value_df, stat_indicators, analysis_cycle, None)
    stat.calculate()
    stat_df = stat.stat_df

    print(f'----------------------- 数据处理完成 -----------------------')

    print('----------------------- 数据保存 -----------------------')
    storage = storage(storage_file_path)
    if not financial_df.empty:
        storage.write_df_to_excel(financial_df, '财务数据')

    if not kline_df.empty:
        storage.write_df_to_excel(kline_df, 'K线数据')

    print('----------------------- 绘图 -----------------------')

    # ———— 管理用财务报表 ———— #
    draw.draw('blpp', '管理用财务报表', '总资产\n总资产=金融性负债+经营性负债+股东入资+利润留存',
              [
                  financial_df.loc[['金融性负债', '经营性负债', '股东入资', '利润留存']],
                  financial_df.loc[['总权益及总负债', '总权益及总负债_yoy']],
                  None,
                  [financial_df.loc[financial_index_comparison_dict['股东入资']],
                   financial_df.loc[financial_index_comparison_dict['利润留存']]],
                  None, True, False
              ])
    try:
        draw.draw('blpp', '管理用财务报表', '总资产\n总资产=经营性资产+金融性资产',
                  [
                      financial_df.loc[['经营性资产', '金融性资产']],
                      financial_df.loc[['经营性资产_yoy', '金融性资产_yoy']],
                      financial_df.loc[['总权益及总负债']],
                      [financial_df.loc[financial_index_comparison_dict['经营性资产']],
                       financial_df.loc[financial_index_comparison_dict['金融性资产']]],
                      None, True, False
                  ])
    except KeyError:
        pass
    try:
        draw.draw('blpp', '管理用财务报表', '总资产\n总资产=流动资产合计+非流动资产合计',
                  [
                      financial_df.loc[['流动资产合计', '非流动资产合计']],
                      financial_df.loc[['流动资产合计_yoy', '非流动资产合计_yoy']],
                      None,
                      [financial_df.loc[financial_index_comparison_dict['流动资产']],
                       financial_df.loc[financial_index_comparison_dict['非流动资产']]],
                      None, True, False
                  ])
    except KeyError:
        pass
    try:
        draw.draw('blpp', '管理用财务报表', '经营/流动\n经营性营运资本=经营性流动资产-经营性流动负债',
                  [
                      financial_df.loc[['经营性流动资产', '经营性流动负债']],
                      financial_df.loc[['经营性营运资本', '经营性营运资本_yoy']],
                      None,
                      [financial_df.loc[financial_index_comparison_dict['经营性流动资产']],
                       financial_df.loc[financial_index_comparison_dict['经营性流动负债']]],
                      None, True, False
                  ])
    except KeyError:
        pass
    try:
        draw.draw('blpp', '管理用财务报表', '经营/长期\n净经营性长期资产=经营性长期资产-经营性长期负债',
                  [
                      financial_df.loc[['经营性长期资产', '经营性长期负债']],
                      financial_df.loc[['净经营性长期资产', '净经营性长期资产_yoy']],
                      None,
                      [financial_df.loc[financial_index_comparison_dict['经营性长期资产']],
                       financial_df.loc[financial_index_comparison_dict['经营性长期负债']]],
                      None, True, False
                  ])
    except KeyError:
        pass
    try:
        draw.draw('blpp', '管理用财务报表', '金融\n净负债=金融性负债-金融性资产',
                  [
                      financial_df.loc[['金融性负债', '金融性资产']],
                      financial_df.loc[['净负债', '净负债_yoy']],
                      None,
                      [financial_df.loc[financial_index_comparison_dict['金融性负债']],
                       financial_df.loc[financial_index_comparison_dict['金融性资产']]],
                      None, True, False
                  ])
    except KeyError:
        pass
    draw.draw('blbl', '管理用财务报表', '净经营资产\n净经营资产=净负债+所有者权益',
              [
                  financial_df.loc[['净经营资产']],
                  financial_df.loc[['净经营资产_yoy']],
                  financial_df.loc[['净负债', '股东权益']],
                  financial_df.loc[['净负债_yoy', '股东权益_yoy']],
              ])
    draw.draw('blpp', '管理用财务报表', '利润',
              [
                  financial_df.loc[['核心利润', '投资利润', '杂项利润']],
                  financial_df.loc[['核心利润_yoy', '投资利润_yoy', '杂项利润_yoy']],
                  None, None, None, True, False
              ])
    draw.draw('blbl', '管理用财务报表', '利润',
              [
                  financial_df.loc[['除税后溢利']],
                  financial_df.loc[['除税后溢利_yoy']],
                  financial_df.loc[['融资成本', '净负债']],
                  financial_df.loc[['除税后溢利', '股东权益']],
              ])
    draw.draw('blll', '管理用财务报表', '管理用财务指标\n权益净利率=净经营资产净利率+杠杆贡献率\n杠杆贡献率='
                                 '经营差异率*净财务杠杆\n经营差异率=净经营资产净利率-税后利息率',
              [
                  financial_df.loc[['权益净利率']],
                  financial_df.loc[['净经营资产净利率', '杠杆贡献率']],
                  financial_df.loc[['经营差异率', '净财务杠杆']],
                  financial_df.loc[['净经营资产净利率', '税后利息率']]
              ])
    draw.draw('blll', '管理用财务报表', '管理用财务指标\n净经营资产净利率=经营净利率*净经营资产周转率\n'
                                 '经营净利率=经营净利润/营业收入\n净经营资产周转率=营业收入/净经营资产',
              [
                  financial_df.loc[['净经营资产净利率']],
                  financial_df.loc[['经营净利率', '净经营资产周转率']],
                  financial_df.loc[['经营净利润', '营运收入']],
                  financial_df.loc[['营运收入', '净经营资产']]
              ])

    # ———— 指标类 ———— #
    draw.draw('blll', '指标', '偿债能力',
              [
                  financial_df.loc[['流动比率', '速动比率', '现金比率', '现金流量比率']],
                  financial_df.loc[['有息负债率', '资产负债率', '长期资本化比率']],
                  financial_df.loc[['利息保障倍数', '现金流量利息保障倍数']],
                  None
              ])
    draw.draw('blll', '指标', '盈利能力',
              [
                  financial_df.loc[['营运收入', '毛利', '核心利润', '除税前溢利', '除税后溢利']],
                  financial_df.loc[['营运收入_yoy', '毛利_yoy', '核心利润_yoy', '除税前溢利_yoy', '除税后溢利_yoy']],
                  financial_df.loc[['毛利率', '核心利润率', '销售利润率', '营业净利率']],
                  None
              ])
    draw.draw('blbl', '指标', '盈利能力',
              [
                  financial_df.loc[['销售及分销费用', '行政开支']],
                  financial_df.loc[['销售费用率', '管理费用率', '毛销差']],
                  financial_df.loc[['期间费用率']],
                  financial_df.loc[['期间费用率_yoy']]
              ])
    draw.draw('blll', '指标', '盈利能力',
              [
                  financial_df.loc[['核心利润获现率', '净现比']],
                  financial_df.loc[['核心利润获现率_yoy', '净现比_yoy']],
                  financial_df.loc[['经营业务现金净额']],
                  None
              ])
    draw.draw('blbl', '指标', '营运能力',
              [
                  financial_df.loc[['营业周期', '现金转换周期']],
                  financial_df.loc[['存货周转天数', '应收票据及应收账款周转天数', '应付票据及应付账款周转天数']],
                  financial_df.loc[['总资产周转天数', '流动资产周转天数', '非流动资产周转天数', '固定资产周转天数']],
                  financial_df.loc[['总资产周转天数_yoy', '流动资产周转天数_yoy', '非流动资产周转天数_yoy', '固定资产周转天数_yoy']]
              ])
    try:
        draw.draw('blll', '指标', '营运能力',
                  [
                      financial_df.loc[['上下游资金占用']],
                      financial_df.loc[['应付款比率', '预收款比率']],
                      financial_df.loc[['单位营收所需的经营性营运资本']],
                      financial_df.loc[['经营性营运资本', '营运收入']]
                  ])
    except KeyError:
        pass
    draw.draw('blbl', '指标', '成长能力\n固定资产对营业收入的推动力=下一年度营业收入增量/本年度固定资产净额增量',
              [
                  financial_df.loc[['固定资产对营业收入的推动力']],
                  financial_df.loc[['营运收入', '物业厂房及设备']],
                  financial_df.loc[['收缩倍数']],
                  None
              ])
    # draw.draw('blbl', '指标', '成长能力',
    #           [
    #               financial_df.loc[['内含增长率', '营业收入_yoy']],
    #               financial_df.loc[['营业净利率', '净经营资产周转率', '利润留存率']],
    #               financial_df.loc[['可持续增长率', '营业收入_yoy']],
    #               financial_df.loc[['营业净利率', '净经营资产周转率', '净经营资产权益乘数', '利润留存率']]
    #           ])
    # draw.draw('bar_overlap_line', '指标', '市值',
    #           [
    #               value_df.loc[['市值']],
    #               value_df.loc[['市值_归一化']],
    #               False,
    #               True
    #           ])
    # draw.draw('bar_overlap_line', '指标', '涨跌',
    #           [
    #               value_df.loc[['涨跌幅']],
    #               value_df.loc[['涨跌幅_归一化']],
    #               False,
    #               True
    #           ])
    # draw.draw('basic_bar', '指标', '估值',
    #           [
    #               value_df.loc[['市盈率', '市净率', '市销率']],
    #               True
    #           ])
    # draw.draw('basic_line', '指标', '估值',
    #           [
    #               value_df.loc[['市盈率_归一化', '市净率_归一化', '市销率_归一化']]
    #           ])
    # draw.draw('basic_bar', '指标', '估值',
    #           [
    #               value_df.loc[['滚动市盈率', '滚动市销率']],
    #               True
    #           ])
    # draw.draw('basic_line', '指标', '估值',
    #           [
    #               value_df.loc[['滚动市盈率_归一化', '滚动市销率_归一化']]
    #           ])
    # draw.draw('basic_bar', '指标', '估值',
    #           [
    #               value_df.loc[['市盈率(平均ROE)', 'roe', 'bvps']],
    #               False
    #           ])
    # draw.draw('basic_line', '指标', '估值',
    #           [
    #               value_df.loc[['市盈率(平均ROE)_归一化']]
    #           ])
    # draw.draw('bar_overlap_line', '指标', '估值',
    #           [
    #               value_df.loc[['股息率']],
    #               value_df.loc[['股息率_归一化']],
    #               False,
    #               True
    #           ])

    draw.render()

    print('----------------------- 完成 -----------------------')
