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
class CleanerToCFFEX:
    """
    清洗金融交易所数据
    """
    def __init__(
            self,
            code: str
    ):
        """
        :param code: 代码
        """
        self.code = code

    # --------------------------
    # 公开 API 方法
    # --------------------------
    @classmethod
    def clean_ccpm(
            cls,
            raw_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        清洗成交持仓排名数据
            -1 席位名称统一，去除括号及其包含文本
            -2 删除 -> 产品ID/会员ID
        :param raw_data: 原始数据
        :return: 清洗后的数据
        """
        # -1 席位名称统一: 匹配英文()和中文（）括号
        pattern = r'$.*?$|\(.*?\)'
        raw_data['会员简称'] = raw_data['会员简称'].str.replace(pattern, '', regex=True)
        # -2 删除产品ID/会员ID
        raw_data.drop(["产品ID", "会员ID"], axis=1, inplace=True)

        return raw_data
