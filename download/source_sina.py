"""
新浪爬虫
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

from constant.type_ import FINANCIAL_SHEET, validate_literal_params


######################################################
class CrawlerToSina:
    """
    新浪网爬虫（财务报表、公告标题、总股本）
    """

    # --------------------------
    # 类属性：配置参数
    # --------------------------
    FINANCIAL_TABLE_INDEX = 13      # 财务报表在HTML中的表格索引
    MAX_RETRIES = 3                 # 最大重试次数
    REQUEST_TIMEOUT = 20            # 请求超时时间

    # URL
    BALANCE_SHEET_URL = 'https://money.finance.sina.com.cn/corp/go.php/vFD_BalanceSheet'
    PROFIT_STATEMENT_URL = 'https://money.finance.sina.com.cn/corp/go.php/vFD_ProfitStatement'
    CASHFLOW_STATEMENT_URL = 'https://money.finance.sina.com.cn/corp/go.php/vFD_CashFlow'
    ANNOUNCEMENT_URL = 'https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletin.php'
    INDUSTRY_URL = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpOtherInfo'
    TOTAL_SHARES_URL = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructureHistory'

    def __init__(
            self,
            code: str,
            user_agent: str,
            start_date: pd.Timestamp | None = None,
            end_date: pd.Timestamp | None = None,
            pause_time: float = 0.5,
    ):
        """
        初始化爬虫参数
        :param code: 股票代码（如：600000）
        :param user_agent: 浏览器代理
        :param pause_time: 请求间隔时间
        :param start_date: (str) 2020
        :param end_date: (str) 2020
        """
        self.code = code

        self.start_date = start_date.strftime('%Y') if start_date is not None else start_date
        self.end_date = end_date.strftime('%Y') if end_date is not None else end_date
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
    ) -> pd.DataFrame:
        """带重试和异常处理的请求包装器"""
        for _ in range(self.MAX_RETRIES):
            try:
                return fetcher(*args, **kwargs)
            except (requests.RequestException, ValueError) as e:
                print(f"请求失败: {e}，剩余重试次数：{self.MAX_RETRIES - _ - 1}")
                time.sleep(self.pause_time * 2)
        return pd.DataFrame()

    def _generate_report_dates(
            self
    ) -> list[str]:
        """生成需要爬取的报告日期列表"""
        """生成需要爬取的报告日期列表"""
        try:
            # 将字符串年份转为整数
            start_year = int(self.start_date)
            end_year = int(self.end_date)

            # 检查年份合理性
            if start_year > end_year:
                print("错误：起始年份不能晚于结束年份")
                return []

            # 生成年份列表
            year_list = [str(y) for y in list(range(start_year, end_year + 1))]
            return year_list

        except ValueError:
            print("错误：输入必须为有效的年份格式（如'2020'）")
            return []

    # --------------------------
    # 爬取财务报表所需的类
    # --------------------------
    def _fetch_financial_table(
            self,
            url: str
    ) -> pd.DataFrame | None:
        """
        通用财务报表获取方法
        :param url: 目标 API 的 URL
        :return: 解析后的财务数据
        """
        try:
            response = self.session.get(
                url,
                timeout=self.REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            tables = pd.read_html(response.text)
            if len(tables) <= self.FINANCIAL_TABLE_INDEX:
                raise ValueError(f"Table index {self.FINANCIAL_TABLE_INDEX} not found")
            return pd.DataFrame(tables[self.FINANCIAL_TABLE_INDEX])
        except Exception as e:
            print(f"获取财务报表失败: {e}")
            return None

    def _get_balance_sheet(
            self,
            year: str
    ) -> pd.DataFrame | None:
        """
        获取资产负债表
        :param year: 年份
        :return: 解析后的资产负债表数据
        """
        return self._fetch_financial_table(
            f"{self.BALANCE_SHEET_URL}/stockid/{self.code}/ctrl/{year}/displaytype/4.phtml"
        )

    def _get_profit_statement(
            self,
            year: str
    ) -> pd.DataFrame | None:
        """
        获取利润表
        :param year: 年份
        :return: 解析后的利润表数据
        """
        return self._fetch_financial_table(
            f"{self.PROFIT_STATEMENT_URL}/stockid/{self.code}/ctrl/{year}/displaytype/4.phtml"
        )

    def _get_cash_flow(
            self,
            year: str
    ) -> pd.DataFrame | None:
        """
        获取现金流量表
        :param year: 年份
        :return: 解析后的现金流量表数据
        """
        return self._fetch_financial_table(
            f"{self.CASHFLOW_STATEMENT_URL}/stockid/{self.code}/ctrl/{year}/displaytype/4.phtml"
        )

    # --------------------------
    # 爬取其他数据所需的类
    # --------------------------
    def _get_announcement_title(
            self,
            page: int
    ) -> pd.DataFrame:
        """
        爬取公告标题
        :param page: 页数
        :return: 公告标题
        """
        try:
            response = self.session.get(
                url=f"{self.ANNOUNCEMENT_URL}?stockid={self.code}&Page={page}",
                timeout=self.REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            # 解析html
            soup = BeautifulSoup(response.text, 'lxml')
            result = soup.select('div.datelist')[0]
            dates, announcement_titles = [], []
            for i, string in enumerate(result.strings):
                if i % 2 == 0:
                    dates.append(string.strip())
                else:
                    announcement_titles.append(string.strip())
            # 整合数据
            df = pd.DataFrame()
            for i, date, title in zip(range(0, 10000, 1), dates, announcement_titles):
                if date and title:
                    df = pd.concat([df, pd.DataFrame({i: {'date': date, 'title': title}}).T])
            return df
        except Exception as e:
            print(f"公告标题获取失败: {str(e)}")
            return pd.DataFrame()

    def _get_industry_classification(
            self
    ) -> pd.DataFrame | str:
        """
        爬取所属行业信息
        """
        try:
            response = self.session.get(
                url=f"{self.INDUSTRY_URL}/stockid={self.code}/menu_num/2.phtml",
                timeout=self.REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')
            return soup.select('.comInfo1 .ct')[3].text
        except Exception as e:
            print(f"行业分类获取失败: {str(e)}")
            return pd.DataFrame()

    def _get_total_shares(
            self
    ) -> pd.DataFrame:
        """
        爬取总股本
        :return: 总股本
        """
        try:
            response = self.session.get(
                url=f"{self.TOTAL_SHARES_URL}/stockid/{self.code}/stocktype/TotalStock.phtml",
                timeout=self.REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            # 解析、整合数据
            soup = BeautifulSoup(response.text, 'lxml')
            data = soup.select('div[align="center"]')
            date, shares = [], []
            for i, d in enumerate(data):
                if len(d) == 1:
                    if i % 2:
                        date.append(d.text)
                    else:
                        shares.append(float(d.text[: -2]) * 10000)
            df = pd.DataFrame({d: s for d, s in zip(date, shares)}, index=['shares']).T
            df.index.name = 'date'
            return df
        except Exception as e:
            print(f"总股本获取失败: {str(e)}")
            return pd.DataFrame()

    # --------------------------
    # 公开 API 方法
    # --------------------------
    @validate_literal_params
    def get_financial_data(
            self,
            type_name: FINANCIAL_SHEET
    ) -> pd.DataFrame:
        """
        获取财务报表数据
        :param type_name: 报表名
        :return: 财务数据
        """
        statements = {
            'BS': self._get_balance_sheet,
            'PS': self._get_profit_statement,
            'CF': self._get_cash_flow
        }

        # 生成报告日期列表
        year_list = self._generate_report_dates()

        # 分批处理日期
        ret = pd.DataFrame()
        for year in year_list:

            # 单次爬取财务数据
            df = self._safe_request(statements[type_name], year)

            # 返回None时，判定个股已退市，无后续数据，退出循环
            if df is None:
                return ret
            else:
                # 合并非空面板数据
                if not df.empty:
                    ret = pd.concat([ret, df], axis=1)

            # 暂停，规避反爬
            time.sleep(self.pause_time)

        return ret

    def get_announcement_title(
            self,
            announcement_max_page: int = 1000
    ) -> pd.DataFrame:
        """
        爬取公告标题
        :return: 公告标题
        """
        ret = pd.DataFrame()
        for page in range(1, announcement_max_page, 1):
            df = self._safe_request(self._get_announcement_title, page)
            # 空值，推出循环
            if df.empty:
                break
            else:
                ret = pd.concat([ret, df])

            # 暂停，规避反爬
            time.sleep(self.pause_time)

        return ret

    def get_industry_classification(
            self
    ) -> pd.DataFrame:
        """
        爬取所属行业信息
        """
        df = self._safe_request(self._get_industry_classification)
        # 暂停，规避反爬
        time.sleep(self.pause_time)

        return df

    def get_total_shares(
            self
    ) -> pd.DataFrame:
        """
        爬取总股本
        """
        df = self._safe_request(self._get_total_shares)
        # 暂停，规避反爬
        time.sleep(self.pause_time)

        return df
