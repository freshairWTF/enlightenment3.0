# encoding:utf-8
"""
生成排序分析文件
"""


import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def data_extraction(folder, file, indicators: list, targets: list):
    """
    数据提取
    :param folder：文件夹名
    :param file：文件名
    :param indicators：财务指标/估值指标
    :param targets：行业/个股
    """
    ret = {}
    for i in indicators:
        ret[i] = {}
        ret[i+'-排序'] = {}

        # 读取文件
        try:
            df = pd.read_excel(folder + f'/{file}.xlsx', sheet_name=i)
            df.rename(columns={'Unnamed: 0': '行业/个股'}, inplace=True)
        except ValueError:
            continue

        # 数据处理
        # 获取最近日期列名
        columns = list(df.columns)
        if i in ['内含增长率', '可持续增长率']:
            last_date = columns[columns.index('均值') - 2]
        else:
            last_date = columns[columns.index('均值') - 1]
        # 排序
        df = df.sort_values(last_date, ascending=False)
        # 索引重置
        df = df.reset_index(drop=True)
        # 截取数据
        intercept_df = pd.concat([df['行业/个股'], df[last_date]], axis=1)
        # 提取数据
        for t in targets:
            industry_data = intercept_df[intercept_df['行业/个股'].str.contains(t)]
            ret[i][t] = industry_data[last_date].values[0]
            ret[i+'-排序'][t] = industry_data.index.tolist()[0] + 1

    return pd.DataFrame(ret)


if __name__ == '__main__':
    # ------------ 参数 ------------ #
    # 存储文件名
    storage_file = '群兴玩具'

    # 行业名/企业名
    industry_name = []
    # individual_name = ['爱美客', '天坛生物', '远兴能源', '宁德时代', '合兴股份', '科博达', '赛力斯', '法拉电子',
    #                    '汇川技术', '三全食品', '广州酒家', '天味食品', '迈瑞医疗', '华测导航', '三花智控', '柏楚电子', '海天精工',
    #                    '科德数控', '东亚机械']
    individual_name = ['群兴玩具']

    # 财务指标/估值指标
    financial_index = ['归母权益净利率', '权益净利率', '有息负债率', '现金转换周期', '毛利率', '核心利润率', '核心利润获现率', '内含增长率',
                       '可持续增长率', '营业收入_yoy', '扩张倍数', '总营业收入', '营业收入', '总净利润', '净利润']
    valuation_indicator = ['滚动市盈率', '核心利润滚动市盈率', '市净率', '滚动市销率', '股息率', '实际收益率', '核心利润实际收益率',
                           '总市值', '市值']

    # 文件路径
    industry_folder = 'support/三级行业24H1'
    individual_folder = 'support/个股24H1'
    storage_folder = 'result_data/排序'

    print('----------------------- 提取数据 -----------------------')
    data = {}
    financial_industry = data_extraction(industry_folder, '财务数据', financial_index, industry_name)
    valuation_industry = data_extraction(industry_folder, '估值数据', valuation_indicator, industry_name)
    financial_individual = data_extraction(individual_folder, '财务数据', financial_index, individual_name)
    valuation_individual = data_extraction(individual_folder, '估值数据', valuation_indicator, individual_name)

    # 数据整合
    financial_df = pd.concat([financial_industry, financial_individual])
    valuation_df = pd.concat([valuation_industry, valuation_individual])

    print('----------------------- 存储数据 -----------------------')
    writer = pd.ExcelWriter(f'{storage_folder}/{storage_file}.xlsx', mode='wb', engine='openpyxl')
    financial_df.to_excel(writer, sheet_name='财务')
    valuation_df.to_excel(writer, sheet_name='估值')
    writer.close()
