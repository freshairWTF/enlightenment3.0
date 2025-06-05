# encoding:utf-8
import os

import pandas as pd
from datetime import datetime, timedelta

from utils import DataLoader, DataStorage
from drawer import DrawByPyecharts

"""
事件分析
"""
"""
回购
    回购金额与指数涨幅
    不同目的回购金额占比
    平均收购力度
    行业分布、排名
    回购对短期股价的影响（1、2、3.。。10）、对长期股价的影响（20、40、60、80、100、。。。240）
    回购规模与长期股价的回归以及平均超额收益
    不同估值公司（市盈率分位数）回购后的股价表现
"""

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

KEY_WORDS = {
    '回购': ['回购', '股份'],
    '增持': ['增持', '计划'],
    '减持': ['减持'],
}
EXCEPT_KEY_WORDS = {
    '回购': ['独立', '意见', '说明', '注销', '债', '结果', '股东', '持股', '权', '实施', '首次', '定向', '质押', '比例',
           '总股本',  '报告', '董事会', '协议', '关联', '通知', '到期', 'B股', 'H股', '仲裁', '关注', '减资', '调整',
           '终止', '完成', '完毕', '进展', '届满', '补充', '情况', '取消', '继续', '延期', '延长', '增信', '装修',
           '司法', '具体', '机制', '补偿', '限制性', '更正', '更新', '超过', '修订', '修正', '担保', '减持', '制度', '规划',
           '管理', '逆回购', '下属', '提示性', '暂停', '恢复', '增加', '用途', '提议', '停牌', '子公司', '出售', '政府',
           '持有', '修改', '事项', '收到', '向', '分公司', '预告'],
    '增持': ['取消价格上限', '完成', '完毕', '进展', '届满', '补充', '终止', '情况', '取消', '继续履行', '调整', '延期',
           '延长'],
    '减持': ['进展', '比例', '达', '过半', '完成', '完毕', '届满', '期满', '未实施', '补充', '终止', '情况', '取消', '继续履行', '调整',
           '延期', '延长', '超过', '不减持', '违规', '承诺', '更正', '独立', '意见', '以下', '公司持有', '提请', '股权', '短线',
           '可能', '债', '结果', '授权', '适时', '更新', '子公司', '分公司', '致歉'],
}
MIN_INTERVAL_DAYS = {
    '回购': 40,
    '增持': 0,
    '减持': 10,
}
"""
！！！事件的自身统计，如发生减持的企业，平均频率，分布情况
！！！事件的企业特征，如谁在减持，他们的估值情况，以及对应的表现情况
"""
if __name__ == '__main__':
    # ----------------------------------------- 初始参数 ----------------------------------------- #
    announcement_title_dir_path = 'announcement_title'
    financial_dir_path = 'cleaned_data/financial_data_from_eastmoney'
    kline_dir_path = 'kline_data'
    listed_company_info = 'support/listed_company_info.xlsx'
    storage_root_path = 'result_data/事件驱动'
    storage_dir_name = '测试'

    # 关键词
    key_words = KEY_WORDS['增持']
    except_key_words = EXCEPT_KEY_WORDS['增持']
    min_interval_days = MIN_INTERVAL_DAYS['增持']

    # 时间区间
    start_date, end_date = '2013-01', '2023-06'
    # 指数代码
    index_code = '000300'
    # 行业分类等级
    industry_level = 1

    print(f'----------------------- 参数初始化 -----------------------')
    # 公告字典
    announcement_title_dict = dict()
    # 财务字典
    financial_dict = dict()
    # k线字典 --> dict k线长度不一，必须使用字典数据类型
    kline_dict = dict()

    # 加载类
    loader = DataLoader
    # 存储实例
    storage = DataStorage(f'{storage_root_path}/{storage_dir_name}')
    # 绘图实例
    draw = DrawByPyecharts(f'{storage_root_path}/{storage_dir_name}/图表.html', ['事件分析'], layout='simple')

    # 年事件发生数 --> list
    date_index = []
    for year in range(int(start_date.split('-')[0]), int(end_date.split('-')[0]) + 1):
        for month in range(1, 13):
            y_m = f'{year}-{month}' if month >= 10 else f'{year}-0{month}'
            if y_m <= end_date:
                date_index.append(y_m)

    # 事件 --> dataFrame
    event_df = pd.DataFrame(columns=date_index, index=['事件触发次数', '上市公司数', '事件触发率']).fillna(0)

    # 行业字典 --> dict
    industry_dict = loader.get_industry_info({'全部': industry_level}, source='SW')
    # 行业事件 --> dataFrame
    industry_event_number_df = pd.DataFrame(columns=date_index, index=list(industry_dict.keys())).fillna(0)

    # 间隔时间
    short_term_interval_day = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    long_term_interval_day = ['20', '40', '60', '80', '100', '120', '140', '160', '180', '200', '220', '240']
    # 超额收益
    excess_return = pd.DataFrame(columns=short_term_interval_day + long_term_interval_day)
    # 估值
    valuation_of_excess_return = pd.DataFrame(columns=long_term_interval_day+['valuation'])
    value_interval = [[0], [0, 20], [20, 40], [40, 60], [60, 80], [80, 100], [100]]
    valuation_df = pd.DataFrame(columns=long_term_interval_day)
    valuation_distribution_df = pd.DataFrame(index=['分布数'])

    # 交易日历 --> dataFrame
    calendar = loader.get_calendar()

    print(f'----------------------- 数据加载、处理中 -----------------------')
    # 指数数据
    d_index_df = loader.get_kline_data(f'{kline_dir_path}/index/day/{index_code}.xlsx', start_date, end_date)
    month_index_df = loader.get_kline_data(f'{kline_dir_path}/index/month/{index_code}.xlsx', start_date, end_date)
    month_index_df = month_index_df.round(2)

    # 获取上市企业数量
    for _, series \
            in pd.read_excel(listed_company_info, sheet_name='company_number', names=['date', '上市公司数']).iterrows():
        if series['date'] in event_df.columns:
            event_df.loc['上市公司数', series['date']] = series['上市公司数']

    # 个股数据
    for root, dir_, files in os.walk(announcement_title_dir_path):
        for file_name in files:

            # if '002422' in file_name:

            code = file_name.split('.')[0]
            print(f'--------------加载{code}数据--------------')
            # 公告数据
            df = pd.read_excel(f'{root}/{file_name}').sort_values('date')
            # 时间过滤
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            # 关键字筛选
            for key_word in key_words:
                df = df[df['title'].str.contains(key_word)]
            # 关键字排除
            for key_word in except_key_words:
                df = df[~df['title'].str.contains(key_word)]
            # 去重
            df = df.drop_duplicates('date')
            # 最小间隔天数过滤
            if min_interval_days and not df.empty:
                delete_date_index = []
                for i, date in enumerate(df['date']):
                    if date not in delete_date_index:
                        for l_date in df['date'].iloc[i+1:]:
                            interval_days = ((datetime.strptime(l_date, '%Y-%m-%d'))
                                             - (datetime.strptime(date, '%Y-%m-%d'))).days
                            if interval_days <= min_interval_days:
                                delete_date_index.append(l_date)
                            else:
                                break
                df = df[~df['date'].isin(delete_date_index)]

            if not df.empty:
                announcement_title_dict[code] = df

                # K线数据/财务数据
                try:
                    kline_dict[code] = loader.get_kline_data(f'{kline_dir_path}/stock/day/{code}.xlsx',
                                                             start_date, end_date)
                    financial_start_date = str(int(start_date.split('-')[0])-1) + '-' + start_date.split('-')[1]
                    financial_dict[code] = loader.get_financial_data(f'{financial_dir_path}/{code}.xlsx',
                                                                     financial_start_date, end_date,
                                                                     'year', key_words=['归属于母公司所有者的净利润', '实收资本']).T
                except FileNotFoundError:
                    del announcement_title_dict[code]

    print('----------------------- 数据统计中 -----------------------')
    for code, df in announcement_title_dict.items():
        print(f'--------------统计{code}数据--------------')
        kline_df = kline_dict[code]
        financial_df = financial_dict[code]

        try:
            industry_name = [k for k, v in industry_dict.items() if code in v][0]
        except IndexError:
            industry_name = ''

        for date in date_index:
            # 事件触发次数/事件触发次数（分行业）
            event = df[df['date'].str.contains(date)]
            if not event.empty:
                event_df.loc['事件触发次数', date] += event.shape[0]
                if industry_name:
                    industry_event_number_df.loc[industry_name, date] += event.shape[0]

        for _, d in df.iterrows():
            # --------------- 期间涨幅/估值 ---------------
            # 定位交易日历（有可能非交易日公告，出现无法定位的情况）
            calendar_index = pd.DataFrame()
            for i in range(10):
                calendar_index = calendar[calendar['date'] == str((datetime.strptime(d['date'], '%Y-%m-%d')
                                                                   + timedelta(days=i)).date())].index
                if not calendar_index.empty:
                    break

            # 再做一重过滤：有可能无法搜到（最新交易日历 < 起始日期）
            if calendar_index.empty:
                print('需更新交易日历')
            else:
                # 起始时间
                s_date = str(calendar.loc[calendar_index]['date'].tolist()[0].date())
                # 计算估值
                try:
                    close = kline_df.loc[s_date, 'close']
                    recent_financial = financial_df[financial_df.index < d['date']]
                    if not recent_financial.empty:
                        recent_series = recent_financial.iloc[-1]
                        valuation = close / (recent_series['归属于母公司所有者的净利润'] / recent_series['实收资本'])
                        valuation_of_excess_return.loc[f'{code}-{s_date}', 'valuation'] = valuation
                except KeyError:
                    pass

                # 统计单天涨幅
                for interval in short_term_interval_day:
                    try:
                        e_date = str(calendar.loc[calendar_index + int(interval)]['date'].tolist()[0].date())
                        stock_increase = kline_df.loc[e_date, 'pctChg']
                        index_increase = d_index_df.loc[e_date, 'pctChg']
                        excess_return.loc[f'{code}-{s_date}', interval] = stock_increase - index_increase
                    except KeyError:
                        """时间周期有可能超出现有的交易日历范畴"""
                        pass
                # 统计阶段时间内的涨幅
                for interval in long_term_interval_day:
                    try:
                        # 个股涨幅
                        e_date = str(calendar.loc[calendar_index + int(interval)]['date'].tolist()[0].date())
                        open_, close = kline_df.loc[s_date, 'open'], kline_df.loc[e_date, 'close']
                        stock_increase = round((close / open_ - 1) * 100, 2)
                        # 指数涨幅
                        open_, close = d_index_df.loc[s_date, 'open'], d_index_df.loc[e_date, 'close']
                        index_increase = round((close / open_ - 1) * 100, 2)
                        # 超额收益
                        excess_return.loc[f'{code}-{s_date}', interval] = stock_increase - index_increase
                        # 估值超额收益
                        valuation_of_excess_return.loc[f'{code}-{s_date}', interval] = stock_increase - index_increase
                    except KeyError:
                        """时间周期有可能超出现有的交易日历范畴"""
                        pass

    # 均值/离散度/偏度/峰度
    mean = round(event_df.loc['事件触发次数'].mean(), 2)
    dispersion = round((event_df.loc['事件触发次数'].std() / event_df.loc['事件触发次数'].mean()), 2)
    skew = round(event_df.loc['事件触发次数'].skew(), 2)
    kurtosis = round(event_df.loc['事件触发次数'].kurtosis(), 2)

    # 事件触发率
    event_df.loc['事件触发率'] = (event_df.loc['事件触发次数'] / event_df.loc['上市公司数'] * 100).round(2)

    # 行业分布特征
    industry_event_total_number = pd.DataFrame(industry_event_number_df.sum(axis=1), columns=['次数']).T\
        .sort_values('次数', axis=1, ascending=False)

    # 超额收益
    excess_return = pd.DataFrame(excess_return.mean().round(2)).T
    short_excess_return = excess_return[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']].rename(index={0: '单日超额收益'})
    long_excess_return = excess_return[['20', '40', '60', '80', '100', '120', '140', '160', '180', '200',
                                        '220', '240']].rename(index={0: '累计超额收益'})

    # 估值
    for i, interval in enumerate(value_interval):
        if i == 0:
            lower_value, upper_value = '-∞', interval[0]
            df = valuation_of_excess_return[valuation_of_excess_return['valuation'] <= upper_value]
        elif i == len(value_interval) - 1:
            lower_value, upper_value = interval[0], '+∞'
            df = valuation_of_excess_return[valuation_of_excess_return['valuation'] > lower_value]
        else:
            lower_value, upper_value = interval[0], interval[1]
            df = valuation_of_excess_return[(valuation_of_excess_return['valuation'] > lower_value)
                                            & (upper_value >= valuation_of_excess_return['valuation'])]
        if not df.empty:
            valuation_df.loc[f'PE：{lower_value}-{upper_value}'] = df.mean()
            valuation_distribution_df.loc['分布数', f'{lower_value}-{upper_value}'] = df.shape[0]

    print('----------------------- 数据存储 -----------------------')
    if announcement_title_dict:
        storage.write_data_to_excel(announcement_title_dict, '原始数据')
        storage.write_df_to_excel(pd.DataFrame({'均值': mean, '离散度': dispersion, '偏度': skew, '峰度': kurtosis},
                                               index=['分布特征']), '统计数据')

    print('----------------------- 绘图 -----------------------')
    draw.bar_overlap_line('事件触发次数', [event_df.loc[['事件触发次数']], month_index_df[['close']].T])
    draw.bar_overlap_line('事件触发率', [event_df.loc[['事件触发率']], month_index_df[['close']].T])
    draw.basic_bar('行业分布特征', [industry_event_total_number])
    draw.basic_bar('短期单日超额收益', [short_excess_return])
    draw.basic_line('长期累计超额收益', [long_excess_return])
    draw.basic_bar('估值分布', [valuation_distribution_df])
    draw.basic_line('长期累计超额收益（估值）', [valuation_df.round(2)])
    draw.render()
