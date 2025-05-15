"""
基本面数据
"""
from dataclasses import dataclass
from typing import Literal
from loguru import logger
from pathlib import Path
from tqdm import tqdm

import multiprocessing
import random
import pandas as pd

from constant.user_agent import USER_AGENT
from constant.download import INDUSTRY_TYPE
from path_config import DataPATH
from source_eastMoney import CrawlerToEastMoney, CleanerToEastMoney
from source_sina import CrawlerToSina
from source_juChao import CrawlerToJuChao
from source_baoStock import BaoStockDownLoader
from data_storage import DataStorage
from data_loader import DataLoader


import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

# ------------------------- company_type -------------------------
# 金融二级行业对应的 company_type
# company_type：1）银行 = 3；2）保险 = 2；3）证券 = 1


#############################################################
class Crawler:
    """爬取基本面数据"""

    # --------------------------
    # 类常量-爬虫配置信息
    # --------------------------
    ROOT = Path("fundamental_data/raw_data")
    DIR_PATH = {
        # 东财（分红融资/财务数据/十大股东）
        "bonus_financing": ROOT / "bonus_financing_from_eastmoney",
        "financial_data": ROOT / "financial_data_from_eastmoney",
        "top_ten_circulating_shareholders": ROOT / "top_ten_circulating_shareholders_from_eastmoney",
        "top_ten_shareholders": ROOT / "top_ten_shareholders_from_eastmoney",

        # 新浪（总股本/公告标题）
        "total_shares": ROOT / "total_shares_from_sina",
        "announcement_title": ROOT / "announcement_title_from_sina",

        # 巨潮（年报/半年报/季度报）
        "annual_report": ROOT / "report_from_juChao",
        "semiannual_report": ROOT / "report_from_juChao",
        "firstQuarter_report": ROOT / "report_from_juChao",
        "thirdQuarter_report": ROOT / "report_from_juChao",
    }
    PAUSE_TIME = {
        "east_money": 5,
        "sina": 3,
        "juChao": 0.5,
    }

    def __init__(
            self,
            data_name: Literal[
                "financial_data", "bonus_financing",
                "top_ten_circulating_shareholders", "top_ten_shareholders",
                "total_shares", "announcement_title",
                "annual_report", "semiannual_report",
                "firstQuarter_report", "thirdQuarter_report"
            ],
            start_date: str,
            end_date: str,
            code: str | None = None,
            filter_mode: Literal["all", "from_code", None] = None,
            industry_info: dict | None = None,
            announcement_max_page: int = 1000
    ):
        """
        初始化爬虫管理器
        :param data_name: 数据类型
        :param start_date: 开始日期（YYYY-MM-DD）
        :param end_date: 结束日期（YYYY-MM-DD）
        :param code: 指定代码，当code_filter=(None, from_code)时，配合使用
        :param filter_mode: 代码过滤规则 (None: 仅指定代码, "all": 全量, "from_code": 从指定代码开始)
        :param industry_info: 行业分类字典 (指定行业下载)
        :param announcement_max_page: 公告标题最大页数
        """
        self.data_name = data_name
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.filter_mode = filter_mode
        self.code = code
        self.industry_info = industry_info
        self.announcement_max_page = announcement_max_page

        self.logger = self.setup_logger()

        # 结束时间判定
        current_date = pd.Timestamp.now()
        if self.end_date > current_date:
            raise TypeError(f"end_date {end_date} > cur_date {current_date}")

        # 读取类
        self.loader = DataLoader
        # 数据源
        self.source = BaoStockDownLoader
        # 存储实例
        self.storage = DataStorage(self.DIR_PATH[data_name])

        # 行业映射字典
        self.industry_map: pd.DataFrame = self.loader.get_industry_codes(
            sheet_name="Total_A", industry_info={"全部": "二级行业"}
        )

    @staticmethod
    def setup_logger() -> logger:
        logger.add("logger/download_fundamental.log", rotation="10 MB", level="INFO")
        return logger

    # --------------------------
    # 通用类
    # --------------------------
    @classmethod
    def get_user_agent(cls):
        """
        反扒机制之一：随机选择上表中的用户代理
        :return: 代理
        """
        return random.choice(USER_AGENT)

    def _modify_start_date(
            self,
            code: str,
            financial: bool = False
    ) -> pd.Timestamp:
        """
        修正起始时间，以K线数据作为判定依据
        :param code: 代码
        :param financial: 是否为财务数据
        :return: 修正后的起始时间
        """
        try:
            df = self.loader.get_kline(
                code=code,
                cycle="original_day",
                adjusted_mode="backward_adjusted"
            )
            # k线起始时间
            start_date = df.index[0]

            if financial:
                start_date -= pd.DateOffset(years=1)

            return start_date if start_date > self.start_date else self.start_date
        except FileNotFoundError:
            return self.start_date

    def _get_code_list(
            self
    ) -> list[str]:
        """获取待下载代码列表"""
        # 指定行业
        if self.industry_info:
            return self.loader.get_industry_codes(sheet_name="Total_A",
                                                  industry_info=self.industry_info,
                                                  return_type="list")
        else:
            # 指定个股
            if self.filter_mode is None:
                return [self.code]
            # 全部个股
            else:
                with self.source() as source:
                    all_codes = sorted(
                        [code[-6:] for code in source.get_all_stock_codes().tolist()]
                    )
                # 过滤
                if self.filter_mode == "from_code":
                    if self.code not in all_codes:
                        raise ValueError(f"Code {self.code} not found in available codes")
                    idx = all_codes.index(self.code) if self.code in all_codes else 0
                    return all_codes[idx:]
            return all_codes

    def _save_data(
            self,
            df: pd.DataFrame,
            file_name: str,
            subset: list[str] | None = None,
            sort_by: list[str] | None = None
    ) -> None:
        """
        数据存储（自动合并、去重、覆盖目标 Sheet）
        :param df: 待存储数据
        :param file_name: 文件名
        :param subset: 去重列名
        :param sort_by: 排序列名
        """
        self.storage.write_df_to_parquet(
            df=df,
            file_name=file_name,
            index=False,
            subset=subset,
            sort_by=sort_by
        )

    # --------------------------
    # 东财专用类
    # --------------------------
    def _modify_company_type(
            self,
            code: str
    ) -> str:
        """
        修正公司类型：适用于东财
        :param code: 代码
        :return: 修正后的 company_type
        """
        df = self.industry_map[self.industry_map["股票代码"].str.contains(code)]
        return "4" if df.empty else INDUSTRY_TYPE.get(df["二级行业"].iloc[0], "4")

    # --------------------------
    # 数据下载方法 东财
    # --------------------------
    def _download_financial(
            self,
            code: str
    ) -> None:
        """
        下载财务数据（东财）
        :param code: 代码

             date   SECUCODE SECURITY_CODE SECURITY_NAME_ABBR  ORG_CODE ORG_TYPE  \
        0  2002-03-31  600001.SH        600001               邯郸钢铁  10002253       通用
        1  2001-12-31  600001.SH        600001               邯郸钢铁  10002253       通用
        2  2001-06-30  600001.SH        600001               邯郸钢铁  10002253       通用
        """
        crawler = CrawlerToEastMoney(
            code=code,
            user_agent=self.get_user_agent(),
            pause_time=self.PAUSE_TIME["east_money"],
            start_date=self._modify_start_date(code=code, financial=True),
            end_date=self.end_date
        )

        try:
            for data_type in ["bs", "ps", "cf"]:
                df = crawler.get_financial_data(data_type, self._modify_company_type(code)).T
                if not df.empty:
                    # 索引命名
                    df.index.name = "date"
                    # 重置索引
                    df = df.reset_index(drop=False)

                    file_name = f"{code}_{data_type}"
                    self._save_data(df, file_name, subset=["date"], sort_by=["date"])
                    self.logger.success(f"下载成功 | Code: {code} | Type: {data_type}")
                else:
                    self.logger.error(f"下载失败 | Code: {code} |  Type: {data_type} | Error: 数据为空值")
        except Exception as e:
            self.logger.error(f"下载失败 | Code: {code} | Error: {str(e)}")

    def _download_top_ten_circulating_shareholders(
            self,
            code: str
    ) -> None:
        """
        下载十大流通股东数据（东财）
        :param code: 代码
        """
        crawler = CrawlerToEastMoney(
            code=code,
            user_agent=self.get_user_agent(),
            pause_time=self.PAUSE_TIME["east_money"],
            start_date=self._modify_start_date(code=code),
            end_date=self.end_date
        )

        try:
            df = crawler.get_top_ten_circulating_shareholders()
            if not df.empty:
                self._save_data(
                    df,
                    code,
                    subset=["END_DATE", "HOLDER_RANK"],
                    sort_by=["END_DATE", "HOLDER_RANK"]
                )
                self.logger.success(f"下载成功 | Code: {code}")
            else:
                self.logger.error(f"下载失败 | Code: {code} | Error: 数据为空值")
        except Exception as e:
            self.logger.error(f"下载失败 | Code: {code} | Error: {str(e)}")

    def _download_top_ten_shareholders(
            self,
            code: str
    ) -> None:
        """
        下载十大流通股东数据（东财）
        :param code: 代码
        """
        crawler = CrawlerToEastMoney(
            code=code,
            user_agent=self.get_user_agent(),
            pause_time=self.PAUSE_TIME["east_money"],
            start_date=self._modify_start_date(code=code),
            end_date=self.end_date
        )

        try:
            df = crawler.get_top_ten_shareholders()
            if not df.empty:
                self._save_data(
                    df,
                    code,
                    subset=["END_DATE", "HOLDER_RANK"],
                    sort_by=["END_DATE", "HOLDER_RANK"]
                )
                self.logger.success(f"下载成功 | Code: {code}")
            else:
                self.logger.error(f"下载失败 | Code: {code} | Error: 数据为空值")
        except Exception as e:
            self.logger.error(f"下载失败 | Code: {code} | Error: {str(e)}")

    def _download_bonus_financing(
            self,
            code: str
    ) -> None:
        """
        下载分红融资（东财）
        :param code: 代码

            SECUCODE SECURITY_CODE SECURITY_NAME_ABBR          NOTICE_DATE  \
        0  002167.SZ        002167               东方锆业  2024-08-22 00:00:00
        1  002167.SZ        002167               东方锆业  2024-04-19 00:00:00
        """
        crawler = CrawlerToEastMoney(
            code=code,
            user_agent=self.get_user_agent(),
            pause_time=self.PAUSE_TIME["east_money"],
            start_date=self._modify_start_date(code=code),
            end_date=self.end_date
        )
        try:
            data_dict = crawler.get_bonus_and_financing()
            for data_type, df in data_dict.items():
                if not df.empty:
                    file_name = f"{code}_{data_type}"
                    self._save_data(df, file_name)
                    self.logger.success(f"下载成功 | Code: {code} | Type: {data_type}")
                else:
                    self.logger.error(f"下载失败 | Code: {code} | Type: {data_type} | Error: 数据为空值")
        except Exception as e:
            self.logger.error(f"下载失败 | Code: {code} | Error: {str(e)}")

    # --------------------------
    # 数据下载方法 新浪
    # --------------------------
    def _download_total_shares(
            self,
            code: str
    ) -> None:
        """
        下载总股本
        :param code: 代码

              date       shares
        0   2007-08-24   37500000.0
        1   2007-09-13   50000000.0
        2   2007-12-13   50000000.0
        3   2008-09-16   50000000.0
        4   2009-05-26   69120000.0
        """
        try:
            crawler = CrawlerToSina(
                code=code,
                user_agent=self.get_user_agent(),
                pause_time=self.PAUSE_TIME["sina"],
            )
            df = crawler.get_total_shares()
            if not df.empty:
                df = df.reset_index(drop=False)
                self._save_data(df, code, subset=["date"], sort_by=["date"])
                self.logger.success(f"下载成功 | Code: {code}")
            else:
                self.logger.error(f"下载失败 | Code: {code} | Error: 数据为空值")
        except Exception as e:
            self.logger.error(f"下载失败 | Code: {code} | Error: {str(e)}")

    def _download_announcement_title(
            self,
            code: str
    ) -> None:
        """
        下载公告标题
        :param code: 代码

              date                                      title
        0   2025-01-25  东方锆业：独立董事关于公司第八届董事会独立董事专门会议2025年第一次会议审核意见
        1   2025-01-25                 东方锆业：关于召开2025年第一次临时股东大会的通知
        2   2025-01-25                      东方锆业：第八届监事会第十七次会议决议公告
        3   2025-01-25                      东方锆业：第八届董事会第十八次会议决议公告
        """
        crawler = CrawlerToSina(
            code=code,
            user_agent=self.get_user_agent(),
            pause_time=self.PAUSE_TIME["sina"]
        )
        try:
            df = crawler.get_announcement_title(announcement_max_page=self.announcement_max_page)
            if not df.empty:
                self._save_data(df, code, sort_by=["date"], subset=["date", "title"])
                self.logger.success(f"下载成功 | Code: {code}")
            else:
                self.logger.error(f"下载失败 | Code: {code} | Error: 数据为空值")
        except Exception as e:
            self.logger.error(f"下载失败 | Code: {code} | Error: {str(e)}")

    # --------------------------
    # 数据下载方法 巨潮
    # --------------------------
    def _download_report(
            self,
            code: str
    ) -> None:
        """
        下载报告
        :param code: 代码

              date                                      title
        0   2025-01-25  东方锆业：独立董事关于公司第八届董事会独立董事专门会议2025年第一次会议审核意见
        1   2025-01-25                 东方锆业：关于召开2025年第一次临时股东大会的通知
        2   2025-01-25                      东方锆业：第八届监事会第十七次会议决议公告
        3   2025-01-25                      东方锆业：第八届董事会第十八次会议决议公告
        """
        crawler = CrawlerToJuChao(
            code=code,
            user_agent=self.get_user_agent(),
            report_type=self.data_name,
            pause_time=self.PAUSE_TIME["juChao"],
            start_date=self._modify_start_date(code=code),
            end_date=self.end_date
        )
        try:
            # 创建文件夹
            dir_path = self.DIR_PATH[self.data_name] / code
            dir_path.mkdir(parents=True, exist_ok=True)
            # 下载
            for year, data in crawler.get_report():
                # 存储
                with open((dir_path / f"{self.data_name}_{year}").with_suffix(".pdf"), "wb") as f:
                    f.write(data)
            self.logger.success(f"下载成功 | Code: {code}")
        except Exception as e:
            self.logger.error(f"下载失败 | Code: {code} | Error: {str(e)}")

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(
            self
    ) -> None:
        """执行爬取任务"""
        codes = self._get_code_list()

        self.logger.info(f"待处理股票数量：{len(codes)}")
        for idx, code in enumerate(codes, 1):
            self.logger.info(f"\n{"#" * 30} 正在处理第 {idx}/{len(codes)} 支股票 [{code}] {"#" * 30}")

            if self.data_name == "financial_data":
                self._download_financial(code)
            elif self.data_name == "top_ten_circulating_shareholders":
                self._download_top_ten_circulating_shareholders(code)
            elif self.data_name == "top_ten_shareholders":
                self._download_top_ten_shareholders(code)
            elif self.data_name == "bonus_financing":
                self._download_bonus_financing(code)
            elif self.data_name == "announcement_title":
                self._download_announcement_title(code)
            elif self.data_name == "total_shares":
                self._download_total_shares(code)
            elif self.data_name in ["annual_report", "semiannual_report",
                                    "firstQuarter_report", "thirdQuarter_report"]:
                self._download_report(code)


#############################################################
@dataclass
class CleanTask:
    code: str
    source_dir: Path
    storage_dir: Path
    industry_map: pd.DataFrame
    data_types: list
    task_type: str


#############################################################
class Cleaner:
    """清洗基本面数据"""

    # --------------------------
    # 类常量-数据地址信息
    # --------------------------
    RAW_ROOT = Path("fundamental_data/raw_data")
    RAW_DIR_PATH = {
        "bonus_financing": RAW_ROOT / "bonus_financing_from_eastmoney",
        "financial_data": RAW_ROOT / "financial_data_from_eastmoney",
        "top_ten_circulating_shareholders": RAW_ROOT / "top_ten_circulating_shareholders_from_eastmoney",
        "top_ten_shareholders": RAW_ROOT / "top_ten_shareholders_from_eastmoney",
    }
    STORAGE_DRI_PATH = {
        "bonus_financing": DataPATH.BONUS_FINANCING_DATA,
        "financial_data": DataPATH.FINANCIAL_DATA,
        "top_ten_circulating_shareholders": DataPATH.TOP_TEN_CIRCULATING_SHAREHOLDERS,
        "top_ten_shareholders": DataPATH.TOP_TEN_SHAREHOLDERS,
    }

    def __init__(
            self,
            data_name: Literal[
                "financial_data", "bonus_financing",
                "top_ten_circulating_shareholders", "top_ten_shareholders"
            ],
            num_processes: int,
            code: str | None = None,
            filter_mode: Literal["all", "from_code", None] = None,
            industry_info: dict | None = None
    ):
        """
        初始化爬虫管理器
        :param data_name: 数据类型
        :param num_processes: 多进程核数
        :param code: 指定代码，当code_filter=(None, from_code)时，配合使用
        :param filter_mode: 代码过滤规则 (None: 仅指定代码, "all": 全量, "from_code": 从指定代码开始)
        :param industry_info: 行业分类字典 (指定行业下载)
        """
        self.data_name = data_name
        self.filter_mode = filter_mode
        self.code = code
        self.industry_info = industry_info
        self.num_processes = num_processes
        self.source_dir = self.RAW_DIR_PATH[self.data_name]
        self.storage_dir = self.STORAGE_DRI_PATH[self.data_name]

        # 日志
        self.logger = self.setup_logger()
        # 读取类
        self.loader = DataLoader
        # 行业映射字典
        self.industry_map: pd.DataFrame = self.loader.get_industry_codes(
            sheet_name="Total_A", industry_info={"全部": "二级行业"}
        )

    @staticmethod
    def setup_logger() -> logger:
        logger.add("logger/clean_fundamental.log", rotation="10 MB", level="INFO")
        return logger

    # --------------------------
    # 通用类
    # --------------------------
    def _get_all_codes(
            self
    ) -> list[str]:
        """获取全部代码"""
        return sorted(list(set(
            [
                file.stem.split('_')[0]
                for file in self.source_dir.glob("*.parquet")
                if file.is_file()
            ]
        )))

    def _get_code_list(
            self
    ) -> list[str]:
        """获取代码列表"""
        # 指定行业
        if self.industry_info:
            return self.loader.get_industry_codes(sheet_name="Total_A",
                                                  industry_info=self.industry_info,
                                                  return_type="list")
        else:
            # 指定个股
            if self.filter_mode is None:
                return [self.code]
            # 全部文件
            else:
                all_codes = self._get_all_codes()
                # 过滤
                if self.filter_mode == "from_code":
                    if self.code not in all_codes:
                        raise ValueError(f"Code {self.code} not found in available codes")
                    idx = all_codes.index(self.code) if self.code in all_codes else 0
                    return all_codes[idx:]
            return all_codes

    @staticmethod
    def _save_data(
            dir_path: Path,
            df: pd.DataFrame,
            file_name: str,
            subset: list[str] | None = None,
            sort_by: list[str] | None = None
    ) -> None:
        """
        数据存储（自动合并、去重、覆盖目标 Sheet）
        :param dir_path: 文件夹存储目录
        :param df: 待存储数据
        :param file_name: 文件名
        :param subset: 去重列名
        :param sort_by: 排序列名
        """
        DataStorage(dir_path).write_df_to_parquet(
            df=df,
            file_name=file_name,
            merge_original_data=False,
            index=False,
            subset=subset,
            sort_by=sort_by
        )

    # --------------------------
    # 东财专用类
    # --------------------------
    @staticmethod
    def _is_financial_company(
            code: str,
            industry_map: pd.DataFrame
    ) -> bool:
        """
        判定公司是否为金融企业（适用于东财）
        :param code: 代码
        :return: 企业类型
        """
        df = industry_map[industry_map["股票代码"].str.contains(code)]
        try:
            return True if df["二级行业"].iloc[0] in INDUSTRY_TYPE.keys() else False
        except IndexError:
            return False

    # --------------------------
    # 数据下载类
    # --------------------------
    def _clean_financial(
            self,
            code: str
    ) -> None:
        """
        清洗财务数据（东财）
        :param code: 代码

             date   SECUCODE SECURITY_CODE SECURITY_NAME_ABBR  ORG_CODE ORG_TYPE  \
        0  2002-03-31  600001.SH        600001               邯郸钢铁  10002253       通用
        1  2001-12-31  600001.SH        600001               邯郸钢铁  10002253       通用
        2  2001-06-30  600001.SH        600001               邯郸钢铁  10002253       通用
        """
        # 暂不支持金融类企业清洗
        if self._is_financial_company(code, self.industry_map):
            return

        for data_type in ["bs", "ps", "cf"]:
            file_name = f"{code}_{data_type}"
            path = (self.source_dir / file_name).with_suffix(".parquet")
            try:
                # 读取原始数据
                raw_df = pd.read_parquet(path)
                # 清洗原始数据
                cleaner = CleanerToEastMoney(
                    code=code,
                    is_financial=self._is_financial_company(code, self.industry_map)
                )
                cleaned_df = cleaner.clean_financial_data(raw_df, data_type)
                # 存储清洗后的数据
                self._save_data(self.storage_dir, cleaned_df, file_name, sort_by=["date"])
                self.logger.success(f"清洗成功 | Code: {code} | Type: {data_type}")
            except Exception as e:
                self.logger.error(f"清洗失败 | Code: {code} | Error: {str(e)}")

    def _clean_bonus_financing(
            self,
            code: str
    ) -> None:
        """
        清洗分红融资（东财）
        :param code: 代码

            SECUCODE SECURITY_CODE SECURITY_NAME_ABBR          NOTICE_DATE  \
        0  002167.SZ        002167               东方锆业  2024-08-22 00:00:00
        1  002167.SZ        002167               东方锆业  2024-04-19 00:00:00
        """
        for data_type in ["lnfhrz", "zfmx", "pgmx"]:
            file_name = f"{code}_{data_type}"
            path = (self.source_dir / file_name).with_suffix(".parquet")
        try:
            # 读取原始数据
            raw_df = pd.read_parquet(path)
            # 清洗原始数据
            cleaner = CleanerToEastMoney(
                code=code,
                is_financial=self._is_financial_company(code, self.industry_map)
            )
            cleaned_df = cleaner.clean_bonus_financing(raw_df, data_type)
            # 存储清洗后的数据
            if cleaned_df.empty:
                self.logger.error(f"下载失败 | Code: {code} | Type: {data_type} | Error: 数据为空值")
            else:
                self._save_data(self.storage_dir, cleaned_df, file_name, sort_by=["date"], subset=["date"])
                self.logger.success(f"清洗成功 | Code: {code} | Type: {data_type}")
        except Exception as e:
            self.logger.error(f"清洗失败 | Code: {code} | Error: {str(e)}")

    def _clean_top_ten_circulating_shareholders(
            self,
            code: str
    ) -> None:
        """
        清洗前十流通股东（东财）
        :param code: 代码
        """
        try:
            # 读取原始数据
            raw_df = pd.read_parquet((self.source_dir / code).with_suffix(".parquet"))
            # 清洗原始数据
            cleaner = CleanerToEastMoney(
                code=code
            )
            cleaned_df = cleaner.clean_ten_circulating_shareholders(raw_df)

            # 存储清洗后的数据
            if cleaned_df.empty:
                self.logger.error(f"下载失败 | Code: {code} | Error: 数据为空值")
            else:
                self._save_data(
                    self.storage_dir,
                    cleaned_df,
                    code,
                    sort_by=["date", "名次"],
                    subset=["date", "名次"]
                )
                self.logger.success(f"清洗成功 | Code: {code}")
        except Exception as e:
            self.logger.error(f"清洗失败 | Code: {code} | Error: {str(e)}")

    def _clean_top_ten_shareholders(
            self,
            code: str
    ) -> None:
        """
        清洗前十股东（东财）
        :param code: 代码
        """
        try:
            # 读取原始数据
            raw_df = pd.read_parquet((self.source_dir / code).with_suffix(".parquet"))
            # 清洗原始数据
            cleaner = CleanerToEastMoney(
                code=code
            )
            cleaned_df = cleaner.clean_ten_shareholders(raw_df)

            # 存储清洗后的数据
            if cleaned_df.empty:
                self.logger.error(f"下载失败 | Code: {code} | Error: 数据为空值")
            else:
                self._save_data(
                    self.storage_dir,
                    cleaned_df,
                    code,
                    sort_by=["date", "名次"],
                    subset=["date", "名次"]
                )
                self.logger.success(f"清洗成功 | Code: {code}")
        except Exception as e:
            self.logger.error(f"清洗失败 | Code: {code} | Error: {str(e)}")

    # --------------------------
    # 多进程方法
    # --------------------------
    @staticmethod
    def _dispatch_task(task: CleanTask) -> None:
        """任务分发入口"""
        if task.task_type == "financial":
            Cleaner._multi_clean_financial(task)
        elif task.task_type == "bonus_financing":
            Cleaner._multi_clean_bonus_financing(task)
        else:
            raise ValueError(f"未知任务类型: {task.task_type}")

    @staticmethod
    def _multi_clean_financial(
            task: CleanTask
    ) -> None:
        """
        清洗财务数据（东财）
             date   SECUCODE SECURITY_CODE SECURITY_NAME_ABBR  ORG_CODE ORG_TYPE  \
        0  2002-03-31  600001.SH        600001               邯郸钢铁  10002253       通用
        1  2001-12-31  600001.SH        600001               邯郸钢铁  10002253       通用
        2  2001-06-30  600001.SH        600001               邯郸钢铁  10002253       通用
        """
        # 暂不支持金融类企业清洗
        if Cleaner._is_financial_company(task.code, task.industry_map):
            return

        try:
            for data_type in task.data_types:
                file_name = f"{task.code}_{data_type}"
                path = (task.source_dir / file_name).with_suffix(".parquet")
                # 读取原始数据
                raw_df = pd.read_parquet(path)
                # 清洗原始数据
                cleaner = CleanerToEastMoney(
                    code=task.code,
                    is_financial=Cleaner._is_financial_company(task.code, task.industry_map)
                )
                cleaned_df = cleaner.clean_financial_data(raw_df, data_type)
                # 存储清洗后的数据
                Cleaner._save_data(task.storage_dir, cleaned_df, file_name, sort_by=["date"])
        except Exception as e:
            print(f"清洗失败 | Code: {task.code} | Error: {str(e)}")

    @staticmethod
    def _multi_clean_bonus_financing(
            task: CleanTask
    ) -> None:
        """
        清洗分红融资（东财）
            SECUCODE SECURITY_CODE SECURITY_NAME_ABBR          NOTICE_DATE  \
        0  002167.SZ        002167               东方锆业  2024-08-22 00:00:00
        1  002167.SZ        002167               东方锆业  2024-04-19 00:00:00
        """
        for data_type in task.data_types:
            file_name = f"{task.code}_{data_type}"
            path = (task.source_dir / file_name).with_suffix(".parquet")
            try:
                # 读取原始数据
                raw_df = pd.read_parquet(path)
                # 清洗原始数据
                cleaner = CleanerToEastMoney(
                    code=task.code,
                    is_financial=Cleaner._is_financial_company(task.code, task.industry_map)
                )
                cleaned_df = cleaner.clean_bonus_financing(raw_df, data_type)
                # 存储清洗后的数据
                if not cleaned_df.empty:
                    Cleaner._save_data(task.storage_dir, cleaned_df, file_name, sort_by=["date"], subset=["date"])
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"清洗失败 | Code: {task.code} | Error: {str(e)}")

    @staticmethod
    def _multi_clean_top_ten_circulating_shareholders(
            task: CleanTask
    ) -> None:
        """清洗前十流通股东（东财）"""

    @staticmethod
    def _multi_clean_top_ten_shareholders(
            task: CleanTask
    ) -> None:
        """清洗前十股东（东财）"""

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(
            self
    ) -> None:
        """执行清洗任务"""
        codes = self._get_code_list()
        self.logger.info(f"待处理股票数量：{len(codes)}")

        for idx, code in enumerate(codes, 1):
            self.logger.info(f"\n{"#" * 30} 正在处理第 {idx}/{len(codes)} 支股票 [{code}] {"#" * 30}")

            if self.data_name == "financial_data":
                self._clean_financial(code)
            elif self.data_name == "bonus_financing":
                self._clean_bonus_financing(code)
            elif self.data_name == "top_ten_circulating_shareholders":
                self._clean_top_ten_circulating_shareholders(code)
            elif self.data_name == "top_ten_shareholders":
                self._clean_top_ten_shareholders(code)

    def multi_run(
            self
    ) -> None:
        """多进程执行清洗任务"""
        codes = self._get_code_list()
        self.logger.info(f"待处理股票数量：{len(codes)}")

        # 根据任务类型生成参数
        if self.data_name == "financial_data":
            task_config = {
                "data_types": ["bs", "ps", "cf"],
                "task_type": "financial"
            }
        elif self.data_name == "bonus_financing":
            task_config = {
                "data_types": ["lnfhrz", "zfmx", "pgmx"],
                "task_type": "bonus_financing"
            }
        else:
            raise ValueError(f"无效任务类型: {self.data_name}")

        # 生成任务列表
        tasks = [
            CleanTask(
                code=code,
                source_dir=self.source_dir,
                storage_dir=self.storage_dir,
                industry_map=self.industry_map,
                data_types=task_config["data_types"],
                task_type=task_config["task_type"]
            )
            for code in codes
        ]

        # 多进程执行
        try:
            with multiprocessing.Pool(self.num_processes) as pool:
                results = pool.imap_unordered(Cleaner._dispatch_task, tasks)
                list(tqdm(results, total=len(codes)))
        except Exception as e:
            self.logger.error(f"流程异常终止: {e!r}", exc_info=True)
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()
