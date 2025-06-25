"""
金融期货交易所爬虫
"""
import random
import time
import requests
import yaml
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup


######################################################
class CrawlerToCFFEX:
    """
    金融交易所
    """

    # --------------------------
    # 类属性：配置参数
    # --------------------------
    MAX_RETRIES = 3                 # 最大重试次数
    REQUEST_TIMEOUT = 20            # 请求超时时间
    FUTURE_START_DATE = {
        "IF": "2010-04-16",
        "IH": "2015-04-16",
        "IC": "2015-04-16",
        "IM": "2022-07-22",
    }

    # URL
    CCPM_URL = 'http://www.cffex.com.cn/sj/ccpm/'

    def __init__(
            self,
            user_agent: str,
            code: str,
            pause_time: float = 0.5,
    ):
        """
        初始化爬虫参数
        :param user_agent: 浏览器代理
        :param code: 股指代码
        :param pause_time: 请求间隔时间
        """
        self.code = code
        self.pause_time = pause_time

        # requests.Session实例
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})

    # --------------------------
    # 通用类
    # --------------------------
    def _safe_request(
            self,
            fetcher,
            *args,
            **kwargs
    ) -> pd.DataFrame | dict:
        """带重试和异常处理的请求包装器"""
        for _ in range(self.MAX_RETRIES):
            try:
                return fetcher(*args, **kwargs)
            except (requests.RequestException, ValueError) as e:
                print(f"请求失败: {e}，剩余重试次数：{self.MAX_RETRIES - _ - 1}")
                time.sleep(self.pause_time * 2)
        return pd.DataFrame()

    # --------------------------
    # 公开 API接口
    # --------------------------
    def get_ccpm(
            self,
            date: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        获取成交持仓排名数据
        :param date: 日期（格式：YYYY-MM-DD）
            PS: IF上市时间 2010-04-16
                IH上市时间 2015-04-16
                IC上市时间 2015-04-16
                IM上市时间 2022-07-22
        :return: 解析后的成交持仓排名数据
        """
        # value-字表 映射字典
        volume_map = {
            "0": "成交量",
            "1": "持买单量",
            "2": "持卖单量"
        }
        result_map = {
            "0": [],
            "1": [],
            "2": []
        }
        # 起始时间不得先于股指上市时间
        if date < self.FUTURE_START_DATE[self.code]:
            raise TypeError(f"日期先于股指上市日期: {self.FUTURE_START_DATE[self.code]} > {date}")

        response = self.session.get(
            url=f"{self.CCPM_URL}/{date[:4]}{date[5:7]}/{date[-2:]}/{self.code}.xml",
            params={"id": random.randint(0, 99)},
            timeout=self.REQUEST_TIMEOUT
        )
        # -2 解析文本
        soup = BeautifulSoup(response.text, 'xml')
        for data in soup.find_all('data'):
            value = data.get('Value')
            volume_name = volume_map[value]
            record = pd.Series(
                {
                    "合约": (data.find('instrumentId') or data.find('instrumentid')).text,
                    "日期": (data.find('tradingDay') or data.find('tradingday')).text,
                    "名次": int(data.find('rank').text),
                    "会员简称": data.find('shortname').text,
                    volume_name: int(data.find('volume').text),
                    f"{volume_name}比上交易日增减": int((data.find('varVolume') or data.find('varvolume')).text),
                    "会员ID": data.find('partyid').text,
                    "产品ID": data.find('productid').text
                }
            )
            result_map[value].append(record)

        try:
            # -1 爬取
            response = self.session.get(
                url=f"{self.CCPM_URL}/{date[:4]}{date[5:7]}/{date[-2:]}/{self.code}.xml",
                params={"id": random.randint(0, 99)},
                timeout=self.REQUEST_TIMEOUT
            )
            # -2 解析文本
            soup = BeautifulSoup(response.text, 'xml')
            for data in soup.find_all('data'):
                value = data.get('Value')
                volume_name = volume_map[value]
                record = pd.Series(
                    {
                        "合约": (data.find('instrumentId') or data.find('instrumentid')).text,
                        "日期": (data.find('tradingDay') or data.find('tradingday')).text,
                        "名次": int(data.find('rank').text),
                        "会员简称": data.find('shortname').text,
                        volume_name: int(data.find('volume').text),
                        f"{volume_name}比上交易日增减": int((data.find('varVolume') or data.find('varvolume')).text),
                        "会员ID": data.find('partyid').text,
                        "产品ID": data.find('productid').text
                    }
                )
                result_map[value].append(record)
            return (
                pd.concat(result_map["0"], axis=1).T,
                pd.concat(result_map["1"], axis=1).T,
                pd.concat(result_map["2"], axis=1).T
            )
        except Exception as e:
            print(f"获取股指成交持仓排名数据失败: {date} | {e}")
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame()
            )


######################################################
class CleanerToEastMoney:
    """
    清洗东财数据
        1、删除特定无用的行
        2、修改财务科目名
        3、转换数据类型
        4、填充Nan（前、后各一位，最后全部0）
        5、去重名（三张表）
    """
    def __init__(
            self,
            code: str,
            is_financial: bool = False
    ):
        """
        :param code: 代码
        :param is_financial: 是否为金融企业
        """
        self.code = code
        self.is_financial = is_financial

        self._load_mappings()

    # --------------------------
    # 清洗财务报表所需的类
    # --------------------------
    def _load_mappings(
            self
    ) -> None:
        """从YAML文件加载科目映射配置"""
        config_path = Path(__file__).parent / "financial_mappings.yaml"
        with open(config_path, encoding='utf-8') as f:
            mappings = yaml.safe_load(f)

        self.bs_mapping = mappings['BALANCE_SHEET_NON_FINANCIAL']
        self.ps_mapping = mappings['PROFIT_STATEMENT_NON_FINANCIAL']
        self.cf_mapping = mappings['CASHFLOW_STATEMENT_NON_FINANCIAL']

    @staticmethod
    def _deduplicate(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """去除重复列（基于报告日期）"""
        return df.loc[~df['REPORT_DATE'].duplicated()]

    @staticmethod
    def _rename_columns(
            df: pd.DataFrame,
            mapping: dict
    ) -> pd.DataFrame:
        """重命名财务科目"""
        return df.rename(columns=mapping)

    @staticmethod
    def _basic_clean(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """基础清洗：类型转换"""
        return df.apply(pd.to_numeric, errors='ignore')

    @staticmethod
    def _basic_fill(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """填充数据：用0填充"""
        return df.fillna(0)

    @staticmethod
    def _filter_columns(
            df: pd.DataFrame,
            cols: list
    ) -> pd.DataFrame:
        """过滤财务科目"""
        cols[:0] = ['date']
        return df[cols]

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def clean_financial_data(
            self,
            raw_data: pd.DataFrame,
            data_type: str
    ) -> pd.DataFrame:
        """
        清洗财务数据
        :param raw_data: 原始数据
        :param data_type: 数据类型
        :return: 清洗后的数据
        """
        mapping = {
            'bs': self.bs_mapping,
            'ps': self.ps_mapping,
            'cf': self.cf_mapping
        }
        df = self._deduplicate(raw_data)
        df = self._rename_columns(df, mapping[data_type])
        df = self._basic_clean(df)
        df = self._basic_fill(df)
        df = self._filter_columns(df, list(mapping[data_type].values()))

        return df

    @staticmethod
    def clean_bonus_financing(
            raw_data: pd.DataFrame,
            data_type: str
    ) -> pd.DataFrame | None:
        """
        清洗分红融资数据
        :param raw_data: 原始数据
        :param data_type: 数据类型
        :return: 清洗后的数据
        """
        if data_type == 'lnfhrz':
            df = raw_data.rename(columns={
                'STATISTICS_YEAR': 'date',
                'TOTAL_DIVIDEND': 'dividend'
            })
            df['date'] = df['date'].astype(str) + '-12-31'
            return df

        if data_type == 'zfmx':
            df = raw_data.rename(columns={
                'NOTICE_DATE': 'date',
                'NET_RAISE_FUNDS': 'net_raise_funds'
            })
            return df

        if data_type == 'pgmx':
            df = raw_data.rename(columns={
                'NOTICE_DATE': 'date',
                'TOTAL_RAISE_FUNDS': 'net_raise_funds'
            })
            return df

    @staticmethod
    def clean_ten_circulating_shareholders(
            raw_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        清洗前十大流通股东
        :param raw_data: 原始数据
        :return: 清洗后的数据
        """
        # 修改列名、去除无效信息列
        rename_dict = {
            'END_DATE': 'date',
            'HOLDER_RANK': '名次',
            'HOLDER_NAME': '股东名称',
            'HOLDER_TYPE': '股东性质',
            'SHARES_TYPE': '股份类型',
            'HOLD_NUM': '持股数',
            'FREE_HOLDNUM_RATIO': '占流通股本持股比例',
            'HOLD_NUM_CHANGE': '增减',
            'CHANGE_RATIO': '变动比例',
        }
        df = raw_data.rename(columns=rename_dict)[list(rename_dict.values())]
        # 修改日期数据类型
        df["date"] = pd.to_datetime(df["date"]).dt.date

        return df

    @staticmethod
    def clean_ten_shareholders(
            raw_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        清洗前十大股东
        :param raw_data: 原始数据
        :return: 清洗后的数据
        """
        # 修改列名、去除无效信息列
        rename_dict = {
                'END_DATE': 'date',
                'HOLDER_RANK': '名次',
                'HOLDER_NAME': '股东名称',
                'SHARES_TYPE': '股份类型',
                'HOLD_NUM': '持股数',
                'HOLD_NUM_RATIO': '占总股本持股比例',
                'HOLD_NUM_CHANGE': '增减',
                'CHANGE_RATIO': '变动比例',
            }
        df = raw_data.rename(columns=rename_dict)[list(rename_dict.values())]
        # 修改日期数据类型
        df["date"] = pd.to_datetime(df["date"]).dt.date

        return df
