"""
东财爬虫
"""

import time
import requests
import yaml
import pandas as pd
from pathlib import Path


######################################################
class CrawlerToEastMoney:
    """
    东方财富网爬虫（财务报表、分红融资）
    """

    # --------------------------
    # 类属性：配置参数
    # --------------------------
    A_BATCH_SIZE = 5                # A股每批处理的时间点数量
    H_BATCH_SIZE = 8                # H股每批处理的时间点数量
    MAX_RETRIES = 3                 # 最大重试次数
    REQUEST_TIMEOUT = 20            # 请求超时时间

    # URL
    BALANCE_SHEET_URL = 'https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/zcfzbAjaxNew'
    PROFIT_STATEMENT_URL = 'https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/lrbAjaxNew'
    CASHFLOW_STATEMENT_URL = 'https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/xjllbAjaxNew'
    BONUS_FINANCING_URL = 'http://emweb.securities.eastmoney.com/PC_HSF10/BonusFinancing/PageAjax?'
    SHAREHOLDER_RESEARCH_URL = "https://datacenter.eastmoney.com/securities/api/data/v1/get"

    def __init__(
            self,
            code: str,
            user_agent: str,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp,
            pause_time: float = 0.5,
    ):
        """
        初始化爬虫参数
        :param code: 股票代码（如：600000）
        :param user_agent: 浏览器代理
        :param pause_time: 请求间隔时间
        :param start_date: 开始日期（格式：YYYY-MM-DD）
        :param end_date: 结束日期（格式：YYYY-MM-DD）
        """
        # 识别市场类型
        if code.endswith(".HK"):
            self.market = "HK"
            self.code = code.replace(".HK", "")
        else:
            self.market = "A"
            self.code = f'SH{code}' if code.startswith('6') else f'SZ{code}'

        self.start_date = start_date
        self.end_date = end_date
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
    # 爬取财务报表所需的类
    # --------------------------
    def _generate_report_dates(
            self
    ) -> list[str]:
        """生成需要爬取的报告日期列表"""
        dates = []
        start_year = int(self.start_date.year)
        end_year = int(self.end_date.year)
        date_list = ['-03-31', '-06-30', '-09-30', '-12-31'] if self.market == 'A' else ['-06-30', '-12-31']

        for year in range(start_year, end_year + 1):
            for quarter in date_list:
                date = pd.to_datetime(f'{year}{quarter}')
                if date <= self.end_date:
                    dates.append(date.strftime('%Y-%m-%d'))
        return sorted(dates)

    def _get_batch_dates(
            self,
            date_list: list[str],
            index: int
    ) -> list[str]:
        """获取当前批次的时间点（适用于爬取三张表数据）"""
        batch = date_list[index:index + self.A_BATCH_SIZE]
        return batch + [batch[-1]] * (self.A_BATCH_SIZE - len(batch)) if len(batch) < self.A_BATCH_SIZE else batch

    def _fetch_financial_statement(
            self,
            url: str,
            dates: list[str],
            company_type: int
    ) -> pd.DataFrame | None:
        """
        通用财务报表获取方法
        :param url: 目标 API 的 URL
        :param dates: 时间列表
        :param company_type: 企业类型
        :return: 解析后的财务数据
        """
        params = {
            'companyType': company_type,
            'reportDateType': 0,
            'reportType': 1,
            'dates': ','.join(dates),
            'code': self.code,
        }
        response = self.session.get(
            url,
            params=params,
            timeout=self.REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        try:
            data = response.json()['data']
            return pd.DataFrame({item['REPORT_DATE'].split()[0]: item for item in data})
        except Exception as e:
            if response.status_code == 200:
                print(f'请求成功，但获取财务报表失败。'
                      f'\n1）参数 company_type = {company_type}'
                      f'\n2）企业期间已退市 {dates}')
            else:
                print(f"获取财务报表失败: {e}")
            return None

    def _get_balance_sheet(
            self,
            dates: list[str],
            company_type: int
    ) -> pd.DataFrame | None:
        """
        获取资产负债表
        :param dates: 时间列表
        :param company_type: 企业类型
        :return: 解析后的资产负债表数据
        """
        return self._fetch_financial_statement(
            url=self.BALANCE_SHEET_URL,
            dates=dates,
            company_type=company_type
        )

    def _get_profit_statement(
            self,
            dates: list[str],
            company_type: int
    ) -> pd.DataFrame | None:
        """
        获取利润表
        :param dates: 时间列表
        :param company_type: 企业类型
        :return: 解析后的利润表数据
        """
        return self._fetch_financial_statement(
            url=self.PROFIT_STATEMENT_URL,
            dates=dates,
            company_type=company_type
        )

    def _get_cash_flow(
            self,
            dates: list[str],
            company_type: int
    ) -> pd.DataFrame | None:
        """
        获取现金流量表
        :param dates: 时间列表
        :param company_type: 企业类型
        :return: 解析后的现金流量表数据
        """
        return self._fetch_financial_statement(
            url=self.CASHFLOW_STATEMENT_URL,
            dates=dates,
            company_type=company_type
        )

    # --------------------------
    # 爬取其他数据所需的类
    # --------------------------
    def _get_bonus_and_financing(
            self,
    ) -> dict[str, pd.DataFrame]:
        """
        获取分红融资数据
        :return: 解析后的分红融资数据
        """
        try:
            params = {
                'code': self.code
            }
            response = self.session.get(
                url=self.BONUS_FINANCING_URL,
                params=params,
                timeout=self.REQUEST_TIMEOUT
            )
            return {k: pd.DataFrame(v) for k, v in response.json().items()}
        except Exception as e:
            print(f"获取分红融资数据失败: {e}")
            return {}

    def _get_top_ten_circulating_shareholders(
            self,
            date: str
    ) -> pd.DataFrame:
        """
        获取十大流通股东
        :return: 十大流通股东明细
        """
        try:
            params = {
                "reportName": "RPT_F10_EH_FREEHOLDERS",
                "columns": "SECUCODE,SECURITY_CODE,END_DATE,HOLDER_RANK,HOLDER_NEW,HOLDER_NAME,HOLDER_TYPE,SHARES_TYPE,HOLD_NUM,FREE_HOLDNUM_RATIO,HOLD_NUM_CHANGE,CHANGE_RATIO",
                "filter": f'(SECUCODE="{self.code}")(END_DATE=\'{date}\')',
                "pageNumber": 1,
                "pageSize": "",
                "quoteColumns": "",
                "sortTypes": 1,
                "sortColumns": "HOLDER_RANK",
                "source": "HSF10",
                "client": "PC",
            }
            response = self.session.get(
                url=self.SHAREHOLDER_RESEARCH_URL,
                params=params,
                timeout=self.REQUEST_TIMEOUT
            )
            return pd.DataFrame(response.json()["result"]["data"])
        except Exception as e:
            print(f"获取十大流通股东数据失败: {date} | {e}")
            return pd.DataFrame()

    def _get_top_ten_shareholders(
            self,
            date: str
    ) -> pd.DataFrame:
        """
        获取十大流通股东
        :return: 十大流通股东明细
        """
        try:
            params = {
                "reportName": "RPT_F10_EH_HOLDERS",
                "columns": "ALL",
                "filter": f'(SECUCODE="{self.code}")(END_DATE=\'{date}\')',
                "pageNumber": 1,
                "pageSize": "",
                "quoteColumns": "",
                "sortTypes": 1,
                "sortColumns": "HOLDER_RANK",
                "source": "HSF10",
                "client": "PC",
            }
            response = self.session.get(
                url=self.SHAREHOLDER_RESEARCH_URL,
                params=params,
                timeout=self.REQUEST_TIMEOUT
            )
            return pd.DataFrame(response.json()["result"]["data"])
        except Exception as e:
            print(f"获取十大股东数据失败: {date} | {e}")
            return pd.DataFrame()

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def get_financial_data(
            self,
            type_name: str,
            company_type: str
    ) -> pd.DataFrame:
        """
        获取财务报表数据
        :param type_name: 报表名
        :param company_type: 企业类型，一般企业为4，其余金融类企业从1-4
        :return: 财务数据
        """
        ret = pd.DataFrame()

        # 生成报告日期列表
        date_list = self._generate_report_dates()
        statements = {
            'bs': self._get_balance_sheet,
            'ps': self._get_profit_statement,
            'cf': self._get_cash_flow
        }
        # 分批处理日期
        for i in range(0, len(date_list), self.A_BATCH_SIZE):
            batch_dates = self._get_batch_dates(date_list, i)
            # 单次爬取财务数据
            df = self._safe_request(statements[type_name], batch_dates, company_type)
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

    def get_bonus_and_financing(
            self,
    ) -> dict:
        """
        获取分红融资数据
        :return: 分红与融资数据
        """
        ret = self._safe_request(self._get_bonus_and_financing)

        # 暂停，规避反爬
        time.sleep(self.pause_time)

        return ret

    def get_top_ten_shareholders(
            self
    ) -> pd.DataFrame:
        """
        获取十大股东数据
        :return: 十大股东数据
        """
        # 例：SH600019 -> 600019.SH
        self.code = self.code[2:] + "." +self.code[:2]
        # 允许空值次数
        empty_time = 3

        result = []
        # 生成报告日期列表
        date_list = self._generate_report_dates()
        # 日期反转，从近期开始爬取
        date_list.sort(reverse=True)

        for date in date_list:
            # 爬取数据
            ret = self._safe_request(
                self._get_top_ten_shareholders,
                date
            )
            if ret.empty:
                if empty_time == 0:
                    break
                else:
                    empty_time -= 1
            # 暂停，规避反爬
            time.sleep(self.pause_time)
            # 添加数据
            result.append(ret)

        return pd.concat(result)

    def get_top_ten_circulating_shareholders(
            self
    ) -> pd.DataFrame:
        """
        获取十大流通股东数据
        :return: 十大流通股东数据
        """
        # 例：SH600019 -> 600019.SH
        self.code = self.code[2:] + "." +self.code[:2]
        # 允许空值次数
        empty_time = 3

        result = []
        # 生成报告日期列表
        date_list = self._generate_report_dates()
        # 日期反转，从近期开始爬取
        date_list.sort(reverse=True)

        for date in date_list:
            # 爬取数据
            ret = self._safe_request(
                self._get_top_ten_circulating_shareholders,
                date
            )
            if ret.empty:
                if empty_time == 0:
                    break
                else:
                    empty_time -= 1
            # 暂停，规避反爬
            time.sleep(self.pause_time)
            # 添加数据
            result.append(ret)

        return pd.concat(result)


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
