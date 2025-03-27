"""
巨潮爬虫
"""

import time
import requests
from typing import Iterator
import pandas as pd


######################################################
class CrawlerToJuChao:
    """
    巨潮资讯网爬虫（报告）
    """

    # --------------------------
    # 类属性：配置参数
    # --------------------------
    CATEGORY = {
        'annual_report': 'category_ndbg_szsh',
        'semiannual_report': 'category_bndbg_szsh',
        'firstQuarter_report': 'category_yjdbg_szsh',
        'thirdQuarter_report': 'category_sjdbg_szsh'
    }

    ORG_ID_URL = 'http://www.cninfo.com.cn/new/information/topSearch/detailOfQuery'
    ANNOUNCEMENT_URL = 'http://www.cninfo.com.cn/new/hisAnnouncement/query'

    def __init__(
            self,
            code: str,
            user_agent: str,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp,
            report_type: str,
            pause_time: float = 0.5
    ):
        """
        初始化爬虫参数
        :param code: 股票代码（如：600000）
        :param user_agent: 浏览器代理
        :param pause_time: 请求间隔时间
        :param start_date: 2020
        :param end_date: 2020
        :param report_type: 'annual_report', 'semiannual_report',
                            'firstQuarter_report', 'thirdQuarter_report'
        """
        self.code = code
        self.start_date = start_date.strftime('%Y')
        self.end_date = end_date.strftime('%Y')
        self.pause_time = pause_time
        self.report_type = report_type

        # requests.Session实例
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})

    # --------------------------
    # 通用类
    # --------------------------
    def _get_org_id(
            self
    ) -> str:
        """
        获取巨潮特有的组织 ID
        :return: 组织 ID
        """
        data = {
            'keyWord': self.code,
            'maxSecNum': 10,
            'maxListNum': 5
        }
        try:
            response = self.session.post(self.ORG_ID_URL, data=data)
            response.raise_for_status()
            org_id = response.json()['keyBoardList'][0]['orgId']
            return org_id
        except (requests.RequestException, KeyError, IndexError) as e:
            raise ValueError(f"Failed to fetch org ID: {e}")

    def _define_parameters(
            self,
            org_id: str,
            year: int
    ) -> dict:
        """
        定义请求参数
        :param org_id: 组织 ID
        :param year: 年份
        :return: 请求参数
        """
        return {
            'pageNum': '1',
            'pageSize': '30',
            'tabName': 'fulltext',
            'column': 'szse',
            'isHLtitle': 'true',
            'sortName': 'time',
            'sortType': 'desc',
            'stock': f'{self.code},{org_id}',
            'seDate': f'{year}-01-01~{year}-12-31',
            'category': self.CATEGORY[self.report_type]
        }

    def _fetch_pdf_urls(
            self,
            headers: dict,
            data: dict
    ) -> Iterator[str]:
        """
        获取 PDF 文件的 URL
        :param headers: 请求头
        :param data: 请求参数
        :return: PDF URL 生成器
        """
        try:
            response = self.session.post(self.ANNOUNCEMENT_URL, headers=headers, data=data)
            response.raise_for_status()
            announcements = response.json().get('announcements', [])

            for item in announcements:
                title = item.get('announcementTitle', '')
                if all(keyword not in title for keyword in ['摘要', '英文', '修订', '更新', '已取消', '正文']):
                    pdf_url = 'https://static.cninfo.com.cn/' + item['adjunctUrl']
                    yield pdf_url
        except (requests.RequestException, KeyError) as e:
            print(f"Failed to fetch PDF URLs: {e}")

    def _download_file(
            self,
            url: str
    ) -> bytes | None:
        """
        下载文件并保存到本地
        :param url: 文件 URL
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Failed to download file: {e}")

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def get_report(
            self
    ) -> Iterator[tuple[int, bytes]]:
        """
        爬取报表文件并存储到本地
        """
        try:
            org_id = self._get_org_id()
            time.sleep(self.pause_time)

            for year in range(int(self.start_date), int(self.end_date) + 1):
                headers = {'Accept': '*/*'}
                data = self._define_parameters(org_id, year)
                for pdf_url in self._fetch_pdf_urls(headers, data):
                    yield year, self._download_file(pdf_url)
                    time.sleep(self.pause_time)
        except Exception as e:
            print(f"An error occurred during crawling: {e}")
