"""
市场偏好捕捉器：特征捕捉
"""

import warnings
import pandas as pd
from collections import Counter

from analyze_parameters import ANALYSIS_FACTOR_SET
from download import PARAMETERS_FOR_FINANCIAL_INDEX_OF_SINA, PARAMETERS_FOR_FINANCIAL_INDEX_OF_EASTMONEY, \
    SENIOR_FINANCIAL_INDEX, PARAMETERS_FOR_VALUATION, PARAMETERS_FOR_STATISTICS
from utils import DataLoader, DataProcessor, DataStorage, SummarizeRowData
from indicators_financial import FinancialIndicators
from statistics import Statistics
from indicators_valuation import Valuation
from drawer import DrawToIndividualAnalysis

warnings.filterwarnings('ignore', category=FutureWarning)


if __name__ == '__main__':
    # ----------------------------------------- 初始参数 ----------------------------------------- #
    # ------------ 资源类参数 ------------ #
    financial_dir_path = 'cleaned_data/financial_data_from_eastmoney'           # 财务数据
    kline_root_path = 'kline_data'                                              # K线数据
    result_root_path = 'result_data'                                            # 分析结果
    data_source = 'eastMoney'                                                   # 数据源 --> eastMoney/sina
    source = 'SW'                                                               # 分类源 --> SW/sina

    # ------------ 一般类参数 ------------ #
    # 分析结果文件夹
    storage_file_name = '市场偏好-测试'
    # 分析时间
    start_date, end_date = '2023-12-31', '2024-03-31'
    # K线指标起时时间：因为每年3、6、9、12月末交易时间截止日不确定，因此通常K线起时时间需要前移,此外，部分量价指标的计算需要更大的提前量
    kline_start_date = '2023-11-30'
    # 分析周期：'quarter'(季度)/year(年)/half(半年)
    analysis_cycle = 'quarter'
    # 去极值方法：mad/sigma
    de_extreme_method = DataProcessor.mad
    # 分位数
    quantile = [0.2, 0.4, 0.6, 0.8]

    # 财务科目/估值方法/统计方法/绘图方法
    financial_indicators = ANALYSIS_FACTOR_SET['财务_市场偏好']
    value_indicators = ANALYSIS_FACTOR_SET['估值_市场偏好']
    stat_indicators = ANALYSIS_FACTOR_SET['统计_市场偏好']

    print('----------------------- 参数初始化 -----------------------')

    # 参数检查
    count = Counter(financial_indicators)
    if [num for num in count.values() if num > 1]:
        exit(f'财务指标参数有重复项')

    # 存储路径
    storage_file_path = f'{result_root_path}/市场偏好/{storage_file_name}'

    # 数据加载类
    loader = DataLoader
    # 数据汇总类
    summarize = SummarizeRowData
    # 存储实例
    storage = DataStorage(storage_file_path)
    # 统计类
    stat = Statistics
    # 绘图实例
    draw = DrawToIndividualAnalysis(storage_file_path, ['市场偏好', 'CV', '线性', 'CV+线性'])

    # 财务科目名
    financial_account_words = []
    # 财务字典：{代码：财务科目/指标 --> dataFrame}
    financial_dict = {}
    # k线字典：{指数/板块/股票：K线数据 --> dataFrame}
    kline_dict = {}
    # 日K字典：{指数/板块/股票：K线数据 --> dataFrame}
    daily_kline_dict = {}
    # 指数k线字典
    index_kline_dict = {}
    # 估值字典
    value_dict = {}
    # 汇总财务字典，以及对应的同比、环比数据：{指数/板块/股票：财务科目/指标 --> dataFrame}
    summarize_financial_dict = {}
    # 汇总估值字典
    summarize_value_dict = {}
    # 量价指标字典
    qp_dict = {}

    # 获取代码
    code_list = loader.get_industry_info({'全部': 1}, merge=True, source=source)

    # 能用的财务指标名 字典
    parameters_for_financial_index = PARAMETERS_FOR_FINANCIAL_INDEX_OF_EASTMONEY if data_source == 'eastMoney' \
        else PARAMETERS_FOR_FINANCIAL_INDEX_OF_SINA
    parameters_for_senior_financial_index = SENIOR_FINANCIAL_INDEX
    # 能用的估值指标名 字典
    parameters_for_valuation = PARAMETERS_FOR_VALUATION
    # 能用的统计方法名 字典
    parameters_for_statistics = PARAMETERS_FOR_STATISTICS

    """
    所有财务数据：financial_indicators
    所有财务数据所需的材料：financial_account_words
    需要计算的财务数据（未加工、预加工）：calculate_financial_index
    """
    # ---------- 获取所需全部需要计算的科目 ----------
    calculate_financial_index = [d for d in financial_indicators if d in parameters_for_financial_index.keys()
                                 or d in parameters_for_senior_financial_index.keys()]

    # ---------- 获取所需全部的材料科目 ----------
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
    # 统计
    for key_word in stat_indicators:
        financial_account_words.extend(parameters_for_statistics[key_word])
    # 去重
    financial_account_words = list(set(financial_account_words))

    print(f'----------------------- 数据加载中，共涉及{len(code_list)}个企业 -----------------------')
    for i, code in enumerate(code_list):
        print(f'———— 第{i + 1}个：{code}正加载数据 ————')
        # 加载财务数据
        try:
            financial_df = loader.get_financial_data(f'{financial_dir_path}/{code}.xlsx', start_date, end_date,
                                                     analysis_cycle, key_words=financial_account_words)
        except FileNotFoundError:
            print(f'未查询到{code}的财务文件')
            continue
        # 完备性检查
        error = loader.complete_inspection(financial_df, financial_account_words, time_length_check=True,
                                           start_date=start_date, end_date=end_date)
        if error or financial_df.empty:
            continue
        else:
            # ———— 加载K线数据 ————
            try:
                kline_df = loader.get_kline_data(f'{kline_root_path}/stock/{analysis_cycle}/{code}.xlsx',
                                                 kline_start_date, end_date, match_financial_date=True)
            except FileNotFoundError:
                try:
                    # 构造K线，波幅为0，收盘价使用最新日K数据替代
                    kline_df = loader.get_kline_data(f'{kline_root_path}/stock/day/{code}.xlsx',
                                                     start_date, end_date, match_financial_date=False)
                    if kline_df.empty:
                        continue
                    else:
                        last_close = kline_df.iloc[:, -1]['close']
                        kline_df = kline_df.iloc[:, :financial_df.shape[1]]
                        kline_df.columns = financial_df.columns
                        kline_df.loc['pctChg'] = 0
                        kline_df.loc['close'] = last_close
                except FileNotFoundError:
                    continue

            # ———— 加载分红数据 ————
            try:
                bonus_df = loader.get_bonus_financing(code)
            except FileNotFoundError:
                print(f'未查询到{code}的分红文件')
                continue

            # ———— 加载日K数据 ————
            try:
                daily_kline_df = loader.get_kline_data(f'{kline_root_path}/stock/day/{code}.xlsx',
                                                       kline_start_date, end_date,
                                                       match_financial_date=False)
            except FileNotFoundError:
                print(f'未查询到{code}的日K文件')
                continue

            # ———— 计算财务指标 ————
            financial = FinancialIndicators(financial_df, bonus_df, calculate_financial_index, data_source)
            financial.calculate()
            financial_df = financial_df.round(2)

            # ———— 计算估值指标 ————
            valuation = Valuation(code, financial_df, kline_df, bonus_df, value_indicators, analysis_cycle)
            valuation.calculate()

            # ———— 计算量价指标 ————
            """
            量价指标尚未写好
            """

            # ———— 数据整合 ————
            kline_dict[code] = kline_df
            financial_dict[code] = financial_df
            value_dict[code] = valuation.valuation
            daily_kline_dict[code] = daily_kline_df

    print(f'----------------------- 数据处理完成，实际涉及{len(financial_dict)}个企业 -----------------------')

    print('----------------------- 汇总数据 -----------------------')
    summarize_financial_dict = summarize.summarize_for_mic(financial_dict, financial_indicators)
    summarize_value_dict = summarize.summarize_for_mic(value_dict, value_indicators)
    pct_chg_dict = summarize.summarize_pct_chg(kline_dict, merge=True)

    """
    量价指标尚未写好
    """

    print('----------------------- 统计数据 -----------------------')
    # 计算分位数
    pct_chg_df = pd.DataFrame.from_dict(pct_chg_dict, orient='index')
    pct_chg_df = pct_chg_df.rename({0: 'pct_Chg'}, axis=1)
    quantile_df = pct_chg_df.quantile(quantile).round(2)

    # 生成分位数字典
    code_classification_dict = {f'<{quantile[0]}': []}
    for i, q in enumerate(quantile):
        if i == len(quantile)-1:
            code_classification_dict[f'>{q}'] = []
        else:
            code_classification_dict[f'{q}-{quantile[i+1]}'] = []

    # 按个股涨跌幅与计算出的分位数对代码进行分类
    for code, series in pct_chg_df.iterrows():
        df = quantile_df[quantile_df >= series[0]].notna()
        if False not in df['pct_Chg'].value_counts().keys():
            code_classification_dict[f'<{quantile[0]}'].append(code)
        elif True not in df['pct_Chg'].value_counts().keys():
            code_classification_dict[f'>{quantile[-1]}'].append(code)
        else:
            f, t = df[df == False].dropna().index[-1], df[df == True].dropna().index[0]
            code_classification_dict[f'{f}-{t}'].append(code)

    # 基本面因子配对
    factor_mean_df = pd.DataFrame(index=list(code_classification_dict.keys()))
    # 迭代各财务指标、估值，与整合的财务df、估值df
    for kw, df in {**summarize_financial_dict, **summarize_value_dict}.items():
        temp = pd.DataFrame(index=list(code_classification_dict.keys()), columns=[kw])
        # 迭代各区间成分股的财务df、估值df
        for range_, code_list in code_classification_dict.items():
            series = df.loc[code_list].iloc[:, -1]
            # 数据清洗：极值处理
            if '_yoy' in kw or '_qoq' in kw:
                series = de_extreme_method(series)
            # 取均值
            temp.loc[range_, kw] = series.astype('float').mean()
        # 数据合并
        factor_mean_df = pd.concat([factor_mean_df, temp], axis=1)

    # 数据展示清晰
    factor_mean_df = factor_mean_df.T.astype('float').round(2)

    print('----------------------- 数据存储 -----------------------')
    if not factor_mean_df.empty:
        storage.write_df_to_excel(factor_mean_df, '原始数据', sheet_name='原始数据')
    if not quantile_df.empty:
        storage.write_df_to_excel(quantile_df, '原始数据', sheet_name='分位数', mode='a')

    print('----------------------- 绘图 -----------------------')
    # 市场偏好
    draw.draw('basic_bar', '市场偏好', '涨跌幅', [quantile_df.T, True])
    for k, v in {'市值': '市值', '营业收入': '营业收入', '营收增速': '营业收入_yoy', '净利润': '净利润',
                 '净利润增速': '净利润_yoy', '权益净利率': '权益净利率', '权益净利率历史分位': '权益净利率_归一化',
                 '市盈率': '市盈率', '市盈率历史分位': '市盈率_归一化', '市净率': '市净率', '市净率历史分位': '市净率_归一化',
                 '市销率': '市销率', '市销率历史分位': '市销率_归一化', '股息率': '股息率',
                 '股息率历史分位': '股息率_归一化'}.items():
        if v in factor_mean_df.index:
            draw.draw('basic_bar', '市场偏好', k, [factor_mean_df.loc[[v]], False])

    # 去nan
    factor_mean_df = factor_mean_df.dropna()
    # 变异系数筛查
    cv = stat.coefficient_of_variation(factor_mean_df)

    # CV动态
    for index, d in cv.items():
        if d > 1:
            draw.draw('basic_bar', 'CV', index, [factor_mean_df.loc[[index]], False])

    # 线性
    for index, series in factor_mean_df.diff(1, axis=1).iterrows():
        series = series.dropna()
        if (len(series) == len(series[series > 0])) or (len(series) == len(series[series < 0])):
            draw.draw('basic_bar', '线性', index, [factor_mean_df.loc[[index]], False])

    # CV+线性
    for index, series in factor_mean_df.diff(1, axis=1).iterrows():
        series = series.dropna()
        if (len(series) == len(series[series > 0])) or (len(series) == len(series[series < 0])):
            if cv.loc[index] > 0.7:
                draw.draw('basic_bar', 'CV+线性', index, [factor_mean_df.loc[[index]], False])

    draw.render()
