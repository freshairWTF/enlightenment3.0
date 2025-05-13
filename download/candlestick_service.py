"""
k线处理类：
    1）下载k线数据；
    2）清洗k线数据；
"""
from functools import partial
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Literal

import time
import pandas as pd
import multiprocessing

from source_baoStock import BaoStockDownLoader, BaoStockCleaner
from data_storage import DataStorage
from data_loader import DataLoader
from constant.download import SHEET_NAME_MAP
from constant.path_config import DataPATH
from constant.index_code import INDEX_CODE
from constant.type_ import KLINE_SHEET, validate_literal_params


#############################################################
class Downloader:
    """k线数据下载"""
    def __init__(
            self,
            dir_path: Path,
            download_object: Literal["index", "stock"] = "index",
            category: str = "day",
            adjust_flag: Literal["1", "2", "3"] = "1",
            start_date: str = "2001-01-01",
            end_date: str = "2025-01-18",
            code: str | None = None,
            filter_mode: Literal["all", "from_code", None] = None,
            industry_info: dict | None = None,
            pause_time: float = 0.1
    ):
        """
        :param dir_path: 数据存储根目录
        :param download_object: 下载类型 (stock/index)
        :param category: 周期类型 (day/week/month)，指数仅支持 day
        :param adjust_flag: 复权模式 (1:后复权, 2:前复权, 3:不复权, 4:指数)
        :param code: 指定代码，当code_filter=(None, from_code)时，配合使用
        :param filter_mode: 代码过滤规则 (None: 仅指定代码, "all": 全量, "from_code": 从指定代码开始)
        :param industry_info: 行业分类字典 (指定行业下载)
        :param pause_time: 暂停时间
        """
        self.download_object = download_object
        self.category = category
        self.adjust_flag = adjust_flag if download_object == "stock" else "1"
        self.start_date = start_date
        self.end_date = end_date
        self.code = code
        self.filter_mode = filter_mode
        self.industry_info = industry_info
        self.pause_time = pause_time

        # 日志
        self.logger = self.setup_logger()
        # 数据源
        self.source = BaoStockDownLoader
        # 获取代码列表
        self.code_list = self._get_code_list()
        # 存储文件夹
        self.dir_path = self._get_storage_path(dir_path, SHEET_NAME_MAP[self.adjust_flag])

    @staticmethod
    def setup_logger() -> logger:
        logger.add("logger/download_candlestick.log", rotation="10 MB", level="INFO")
        return logger

    # --------------------------
    # 通用类
    # --------------------------
    def _get_storage_path(
            self,
            dir_path: Path,
            adjusted_mode: str
    ) -> Path:
        """生成存储文件夹路径"""
        if self.download_object == "stock":
            path = dir_path / "stock" / f"original_{self.category}" / adjusted_mode
        else:
            path = dir_path / "index" / f"original_{self.category}"

        return path

    def _get_code_list(
            self
    ) -> list[str]:
        """获取待下载代码列表"""
        # 个股
        if self.download_object == "stock":
            # 指定行业
            if self.industry_info:
                return DataLoader.get_industry_codes(sheet_name="Total_A",
                                                      industry_info=self.industry_info,
                                                      return_type="list")
            else:
                # 指定个股
                if self.filter_mode is None:
                    return [self.code]
                # 全部个股
                else:
                    with self.source() as source:
                        all_codes = source.get_all_stock_codes().tolist()
                        all_codes.sort()
                    # 过滤
                    if self.filter_mode == "from_code":
                        idx = all_codes.index(self.code) if self.code in all_codes else 0
                        return all_codes[idx:]
                return all_codes
        # 指数
        else:
            return INDEX_CODE

    @staticmethod
    def _save_data(
            dir_path: Path,
            df: pd.DataFrame,
            code: str
    ) -> None:
        """数据存储（自动合并、去重、覆盖目标 Sheet）"""
        DataStorage(dir_path).write_df_to_parquet(
            df=df,
            file_name=code,
            index=False,
            merge_original_data=True,
            subset=["date"],
            sort_by=["date"]
        )


    # --------------------------
    # 下载主逻辑
    # --------------------------
    def _download_single(
            self, source,
            code: str
    ) -> bool:
        """下载单个代码数据"""
        try:
            df = source.get_kline_data(
                code=code,
                start_date=self.start_date,
                end_date=self.end_date,
                frequency=self.category[0],
                adjust_flag=self.adjust_flag,
                index=(self.download_object == "index")
            )
            if not df.empty:
                self._save_data(self.dir_path, df, code[-6:])
                self.logger.success(f"下载成功 | Code: {code}")
                return True
            else:
                self.logger.warning(f"数据为空 | Code: {code}")
                return True
        except Exception as e:
            self.logger.error(f"下载失败 | Code: {code} | Error: {str(e)}")
            return False

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(self):
        """执行批量下载"""
        self.logger.info(f"开始下载 | 总数: {len(self.code_list)} | 类型: {self.download_object}")

        with self.source() as source:
            for idx, code in enumerate(self.code_list, 1):
                self.logger.info(f"开始下载 | {idx}: {code} | 类型: {self.download_object}")
                result = self._download_single(source, code)
                if result:
                    time.sleep(self.pause_time)
                else:
                    exit(0)



#############################################################
class Cleaner:
    """数据清洗（支持股票/指数数据多进程清洗）"""

    @validate_literal_params
    def __init__(
            self,
            clean_object: Literal["index", "stock"] = "index",
            num_processes: int = 10,
            adjust_mode: KLINE_SHEET = "backward_adjusted",
            code: str | None = None,
            filter_mode: Literal["all", "from_code", None] = None,
            industry_info: dict | None = None,
    ):
        """
        :param clean_object: 处理类型 "stock"（个股） 或 "index"（指数）
        :param num_processes: 进程池大小
        :param filter_mode: 代码过滤规则 (None: 仅指定代码, "all": 全量, "from_code": 从指定代码开始)
        :param code: 默认股票/指数代码
        :param adjust_mode: 复权类型 "1"（后复权）/ "2"（前复权）/ "3"（不复权）
        :param industry_info: 行业分类字典（可选）
        """
        self.clean_object = clean_object
        self.code = code
        self.filter_mode = filter_mode
        self.industry_info = industry_info if clean_object == "stock" else None
        self.num_processes = num_processes

        # 日志
        self.logger = self.setup_logger()
        # 复权模式
        self.adjust_mode = adjust_mode if clean_object == "stock" else None
        # 存储文件夹
        self.dir_path = self._get_storage_path()
        # 数据源
        self.cleaner = (
            BaoStockCleaner.cleaning_stock if clean_object == "stock"
            else BaoStockCleaner.cleaning_index
        )
        # 获取代码列表
        self.code_list = self._get_code_list()

    @staticmethod
    def setup_logger() -> logger:
        logger.add("logger/clean_candlestick.log", rotation="10 MB", level="INFO")
        return logger

    # --------------------------
    # 通用类
    # --------------------------
    def _get_storage_path(
            self
    ) -> Path:
        """生成存储根目录"""
        return (
            DataPATH.STOCK_KLINE_DATA if self.clean_object == "stock"
            else DataPATH.INDEX_KLINE_DATA
        )

    def _get_code_list(
            self
    ) -> list:
        """获取待处理代码列表"""
        # 个股
        if self.clean_object == "stock":
            # 指定行业
            if self.industry_info:
                return DataLoader.get_industry_codes(sheet_name="Total_A",
                                                      industry_info=self.industry_info,
                                                      return_type="list")
            # 指定个股
            if self.filter_mode is None:
                return [self.code]
            # 根目录内已有的全部文件
            else:
                original_day_path = self.dir_path / "original_day" / str(self.adjust_mode)
                if not original_day_path.exists():
                    raise FileNotFoundError(f"原始数据目录不存在: {original_day_path}")
                code_list = [f.stem for f in original_day_path.iterdir() if f.is_file()]
                code_list.sort()
                # 过滤
                if self.filter_mode == "from_code":
                    idx = code_list.index(self.code) if self.code in code_list else 0
                    return code_list[idx:]
                return code_list
        # 指数
        else:
            # 根目录内已有的全部文件
            original_day_path = self.dir_path / "original_day"
            if not original_day_path.exists():
                raise FileNotFoundError(f"原始数据目录不存在: {original_day_path}")
            code_list = [f.stem for f in original_day_path.iterdir() if f.is_file()]
            code_list.sort()
            return code_list

    @staticmethod
    def _save_data(
            dir_path: Path,
            df: pd.DataFrame,
            code: str
    ) -> None:
        """
        数据存储（自动合并、去重、覆盖目标 Sheet）
                 date       code        open        high         low       close  \
        0  2001-01-05  sz.000001  420.977539  425.610016  408.817261  411.133484
        1  2001-01-12  sz.000001  411.133484  440.086548  400.999908  428.505341
        2  2001-01-19  sz.000001  428.505341  437.191254  408.527710  433.716888
        3  2001-02-09  sz.000001  434.295959  437.191254  405.342896  416.634583
        4  2001-02-16  sz.000001  416.924103  419.819427  406.790527  410.843964

             preclose    volume        amount  adjust_factor    pctChg  real_pctChg  \
        0  420.398468  20324033  2.915377e+08      28.953063 -0.023384    -0.022039
        1  411.133484  27689886  4.004873e+08      28.953063  0.042254     0.042254
        2  428.505341  21621645  3.199792e+08      28.953064  0.012162     0.012162
        3  433.716888  10454562  1.517747e+08      28.953063 -0.040667    -0.039386
        4  416.634583   8213919  1.170088e+08      28.953064 -0.014583    -0.013899

               turn  tradestatus  isST  adjustflag
        0  1.459867            1     0           1
        1  1.988955            1     0           1
        2  1.553075            1     0           1
        3  0.750947            1     0           1
        4  0.590003            1     0           1
        """
        DataStorage(dir_path).write_df_to_parquet(
            df=df,
            file_name=code,
            index=False,
            merge_original_data=False,
            subset=["date"],
            sort_by=["date"]
        )

    # --------------------------
    # 清洗主逻辑
    # --------------------------
    @staticmethod
    def _clean_task(
            cleaner,
            code: str,
            dir_path: Path,
            adjusted_mode: str | None
    ) -> None:
        """清洗任务（静态方法，用于多进程）"""
        try:
            kline_dict = cleaner(
                dir_=dir_path,
                code=code,
                adjusted_mode=adjusted_mode
            )
            for dir_name, df in kline_dict.items():
                path = (
                    dir_path / dir_name / adjusted_mode if adjusted_mode
                    else dir_path / dir_name)
                Cleaner._save_data(path, df, code)
        except Exception as e:
            print(f"清洗/存储失败 | Code: {code} | Error: {str(e)}")

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(self) -> None:
        """启动数据处理管道"""
        self.logger.info("============== 启动数据清洗流程 ==============")
        self.logger.info(f"存储路径: {self.dir_path}")
        self.logger.info(f"处理数量: {len(self.code_list)} 条代码")

        # 创建部分函数固定参数
        task_func = partial(
            Cleaner._clean_task,
            self.cleaner,                     # 绑定到第1个参数: cleaner
            dir_path=self.dir_path,                 # 关键字绑定第3个参数
            adjusted_mode=self.adjust_mode          # 关键字绑定第4个参数
        )

        # 多进程执行
        try:
            with multiprocessing.Pool(self.num_processes) as pool:
                results = pool.imap_unordered(task_func, self.code_list)
                list(tqdm(results, total=len(self.code_list)))
        except Exception as e:
            self.logger.error(f"流程异常终止: {e!r}", exc_info=True)
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()

    def debug(self) -> None:
        """debug"""
        self._clean_task(self.cleaner, self.code_list[0], self.dir_path, self.adjust_mode)
