import os
import pandas as pd

from pathlib import Path
from typing import get_args

from data_storage import DataStorage
from data_loader import DataLoader
from constant.path_config import DataPATH

from constant.type_ import CYCLE

"""
根据月度K线，获取各个企业的上市、退市时间，再统计汇总各月度个数，
"""


##############################################################
class SupportDataUpdater:
    """更新支持数据（交易日历、上市数、行业分类表）"""

    def __init__(
            self,
            start_date: str,
            end_date: str,
            get_listed_code: bool = True
    ):
        """
        初始化处理器
        :param start_date: 开始日期（YYYY-MM-DD）
        :param end_date: 结束日期（YYYY-MM-DD）
        :param end_date: 获取上市代码
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) + pd.DateOffset(months=1)

        self.loader = DataLoader
        self.storage = DataStorage(DataPATH.SUPPORT_DATA)

        # 配置路径
        self.stock_kline_path = DataPATH.STOCK_KLINE_DATA

        self.listed_code_dict = self._get_listed_code() if get_listed_code else {}

    # --------------------------
    # 通用类方法
    # --------------------------
    @staticmethod
    def _get_differences(
            existing_codes: set,
            reference_codes: set
    ) -> tuple[set, set]:
        """获取代码差异"""
        to_remove = existing_codes - reference_codes
        to_add = reference_codes - existing_codes

        return to_remove, to_add

    def _get_listed_code(
            self
    ) -> dict[str, list[str]]:
        """获取各月度上市公司代码（根据月K数据）"""
        # -1 获取月度时间区间
        date_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq="ME"
        ).strftime("%Y-%m")

        # -2 初始化 上市企业代码字典
        listed_code_dict = {date: [] for date in date_range}

        # -3 遍历K线文件统计（月度）
        cycle = "month"
        for root, _, files in os.walk(self.stock_kline_path / cycle):
            for file in files:
                # 获取月k数据
                code = Path(file).stem
                kline_data = self.loader.get_kline(
                    code=code,
                    cycle="month",
                    adjusted_mode="backward_adjusted"
                )
                # 获取上市日期
                listed_dates = kline_data.index.strftime("%Y-%m")
                # 加入字典
                for date in listed_dates:
                    try:
                        listed_code_dict[date].append(code)
                    except KeyError:
                        continue

        return listed_code_dict

    def _save_data(
            self,
            df: pd.DataFrame,
            file_name: str | Path,
            sheet_name: str | None = None
    ) -> None:
        """
        数据存储（追加不合并原数据，无需索引）
        """
        self.storage.write_df_to_excel(
            df,
            file_name,
            sheet_name=sheet_name,
            mode="a",
            index=False,
            merge_original_data=False
        )

    # --------------------------
    # 行业分类相关 私有方法
    # --------------------------
    def _check_duplicates(self) -> None:
        """检查行业分类表重复项"""
        industry_df = self.loader.get_industry_codes(
            sheet_name="A",
            industry_info={"全部": "一级行业"},
            return_type="dataframe"
        )

        # 检查代码重复
        code_dupes = industry_df[industry_df.duplicated("股票代码")]
        self._save_data(code_dupes, DataPATH.INDUSTRY_CLASSIFICATION_UPDATER, "code_dupes")

        # 检查简称重复
        name_dupes = industry_df[industry_df.duplicated("公司简称")]
        self._save_data(name_dupes, DataPATH.INDUSTRY_CLASSIFICATION_UPDATER, "name_dupes")

    def _check_newest_info(self) -> None:
        """检查最新信息"""
        # 行业分类表现存代码
        table_codes = self.loader.get_industry_codes(
            sheet_name="A",
            industry_info={"全部": "一级行业"},
            return_type="list"
        )

        # 最新上市代码
        last_key = list(self.listed_code_dict.keys())[-1]
        newest_listed_codes = self.listed_code_dict[last_key]

        # 获取代码差异
        to_remove, to_add = self._get_differences(set(table_codes), set(newest_listed_codes))

        self._save_data(
            pd.DataFrame(to_remove, columns=["code"]),
            DataPATH.INDUSTRY_CLASSIFICATION_UPDATER,
            sheet_name="A_to_remove"
        )
        self._save_data(
            pd.DataFrame(to_add, columns=["code"]),
            DataPATH.INDUSTRY_CLASSIFICATION_UPDATER,
            sheet_name="A_to_add"
        )

    def _check_past_info(self) -> None:
        """检查过去信息"""
        # 行业分类表全部代码
        table_codes = self.loader.get_industry_codes(
            sheet_name="Total_A",
            industry_info={"全部": "一级行业"},
            return_type="list"
        )
        all_codes = set(sum([code for code in self.listed_code_dict.values()], []))

        # 获取代码差异
        to_remove, to_add = self._get_differences(set(table_codes), set(all_codes))
        self._save_data(
            pd.DataFrame(to_remove, columns=["code"]),
            DataPATH.INDUSTRY_CLASSIFICATION_UPDATER,
            sheet_name="total_A_to_remove"
        )
        self._save_data(
            pd.DataFrame(to_add, columns=["code"]),
            DataPATH.INDUSTRY_CLASSIFICATION_UPDATER,
            sheet_name="total_A_to_add"
        )

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def listed_nums(self) -> None:
        """更新月度上市公司数量（根据月K数据）"""
        # 统计各月度上市公司数量
        listed_nums = pd.DataFrame(
            {k: len(v) for k, v in self.listed_code_dict.items()},
            index=["listed_nums"]
        ).T.reset_index(names="date")

        self._save_data(listed_nums, DataPATH.LISTED_NUMS)

    def trading_calendar(self) -> None:
        """更新交易日历"""
        for cycle in get_args(CYCLE):
            calendar = self.loader.get_index_kline(
                code="000001",
                cycle=cycle,
            ).index.to_frame(index=False)

            self._save_data(calendar, DataPATH.TRADING_CALENDAR, cycle)

    def industry_classification(self) -> None:
        """生成行业分类更新表（退市、新上市、去重）"""
        # 去重
        self._check_duplicates()
        # 退市、新上市
        self._check_newest_info()
        self._check_past_info()

    @staticmethod
    def run(tasks):
        """
        执行任务调度
        :param tasks: 需要执行的任务字典 {方法名: True/False}
        """
        for method, should_run in tasks.items():
            if should_run:
                print(f"------------开始运行程序：{method.__name__}------------")
                method()
