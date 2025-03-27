"""
k线监控业务层
"""

import pandas as pd
from loguru import logger
from pathos.multiprocessing import ProcessPool as Pool

from constant.path_config import DataPATH
from constant.type_ import CYCLE, INDUSTRY_SHEET, KLINE_SHEET, validate_literal_params
from data_storage import DataStorage
from data_loader import DataLoader
from kline_metrics import KLineMetrics

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


###############################################################
class KlineMonitor:

    @validate_literal_params
    def __init__(
            self,
            params,
            start_date: str,
            end_date: str,
            storage_dir_name: str,
            target_info: dict[str, str] | list | str,
            index_code: str,
            cycle: CYCLE = "day",
            code_range: INDUSTRY_SHEET = "A",
            kline_adjust: KLINE_SHEET = "backward_adjusted",
            draw_filter: bool = False,
            processes_nums: int = 1
    ):
        """
        :param cycle: 周期
        :param params: 指定参数
        :param start_date: 起始时间
        :param end_date: 结束时间
        :param storage_dir_name: 存储文件夹名
        :param target_info: 行业信息 -1 {"白酒": 3} -2 ["688305", "601882"] -3 "601882"
        :param kline_adjust: k线复权方法
        :param draw_filter: 绘图过滤
        :param index_code: 指数代码
        :param code_range: 代码范围
        :param processes_nums: 多进程数
        """
        self.PARAMS = params
        self.cycle = cycle
        self.start_date = start_date
        self.end_date = end_date
        self.result_data_path = DataPATH.KLINE_MONITOR_RESULT / storage_dir_name
        self.target_info = target_info
        self.kline_adjust = kline_adjust
        self.draw_filter = draw_filter
        self.index_code = index_code
        self.code_range = code_range
        self.processes_nums = processes_nums

        # --------------------------
        # 初始化配置参数
        # --------------------------
        # 工具类
        self.loader = DataLoader

        # 数据存储容器
        self.kline = {}
        self.index_kline: pd.DataFrame | None = None

        # 初始化其他配置
        self._initialize_config()

        # 日志
        self.logger = self._setup_logger()

    # --------------------------
    # 初始化方法
    # --------------------------
    def _initialize_config(
            self
    ) -> None:
        """初始化配置参数"""
        # --------------------------
        # 路径参数
        # --------------------------
        # 数据路径
        self.kline_stock_path = DataPATH.STOCK_KLINE_DATA

        # --------------------------
        # 其他参数
        # --------------------------
        # 目标代码
        self.target_codes = self._get_target_codes()
        # 目标行业字典
        self.industry_codes: pd.DataFrame = self._get_industry_codes()
        # 方法映射
        self.FUNCTION_MAP = self._load_function()
        # 行业统计个数
        self.slope_critical = 45

    def _setup_logger(
            self
    ) -> logger:
        """配置日志记录器"""
        logger.add(f"{self.result_data_path}/日志.log", rotation="10 MB", level="INFO")
        return logger

    def _get_target_codes(
            self
    ) -> list[str]:
        """获取目标代码列表"""
        # -1 字典
        if isinstance(self.target_info, dict):
            return self.loader.get_industry_codes(
                sheet_name=self.code_range,
                industry_info=self.target_info,
                return_type="list"
            )
        # -2 列表
        elif isinstance(self.target_info, list):
            return self.target_info
        else:
            raise TypeError(f"量价监控不支持该数据类型 {type(self.target_info)}")

    def _get_industry_codes(
            self
    ) -> pd.DataFrame:
        """获取目标行业字典"""
        return (
            self.loader.get_industry_codes(
                sheet_name=self.code_range,
                industry_info={"全部": "三级行业"},
                return_type='dataframe'
            )
            .assign(**{"股票代码": lambda df: df["股票代码"].str.split(".").str[0]})
        )

    def _load_function(
            self
    ) -> dict:
        """获取指标方法名（财务、估值、k线、统计）"""
        return {
            "KLINE": self.loader.load_yaml_file(
                DataPATH.KLINE_METRICS, "FUNCTION_OF_KLINE"
            )
        }

    # --------------------------
    # 数据处理流程
    # --------------------------
    def _load_data(
            self
    ) -> None:
        """加载数据"""
        self.index_kline = self._load_index_data()

        # 使用多进程池并行处理
        with Pool(self.processes_nums) as pool:
            results = pool.map(self._process_single_company, self.target_codes)

        # 合并处理结果到数据容器
        self._unpack_results(results)

    def _unpack_results(
            self,
            results: list[dict[str, str | pd.DataFrame]]
    ) -> None:
        """解包数据"""
        # 合并处理结果到数据容器
        for result in results:
            if result is None:
                continue
            code = result["code"]
            if "error" in result:
                self.logger.error(result["error"])
                continue
            # 更新数据容器
            self.kline[code] = result["kline"]

    def _process_single_company(
            self,
            code: str
    ) -> dict[str, str | pd.DataFrame]:
        """处理单个企业数据"""
        try:
            return {
                "code": code,
                "kline": self._calculate_kline(
                    self._load_kline_data(code)
                )
            }
        except Exception as e:
            return {
                "code": code,
                "error": {str(e)}
            }

    # --------------------------
    # 数据方法
    # --------------------------
    def _load_index_data(
            self
    ) -> pd.DataFrame:
        """加载指数数据"""
        return self.loader.get_index_kline(
            code=self.index_code,
            cycle=self.cycle,
            start_date=self.start_date,
            end_date=self.end_date
        )

    def _load_kline_data(
            self,
            code: str
    ) -> pd.DataFrame:
        """加载k线数据"""
        return self.loader.get_kline(
            code=code,
            cycle=self.cycle,
            adjusted_mode=self.kline_adjust,
            start_date=self.start_date,
            end_date=self.end_date
        )

    def _calculate_kline(
            self,
            kline: pd.DataFrame,
            index_kline: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """计算量价指标"""
        calculator = KLineMetrics(
            kline_data=kline,
            index_data=index_kline,
            cycle=self.cycle,
            methods=self.PARAMS.kline.kline,
            function_map=self.FUNCTION_MAP["KLINE"]
        )
        calculator.calculate()
        return calculator.metrics.round(4)

    # --------------------------
    # 聚合、筛选方法
    # --------------------------
    def _aggregate_into_panel_data(
            self
    ) -> None:
        """处理数据聚合"""
        self.kline = self.__aggregate_into_panel_data(self.kline)
        self.kline = self._get_valid_data(self.kline, "slope")
        self.kline = self._add_industry(self.kline)
        self.kline = self._reset_index(self.kline)

    @classmethod
    def __aggregate_into_panel_data(
        cls,
        data: dict[str, pd.DataFrame],
        metrics: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        将数据按指标聚合为面板数据 {metrics: DataFrame}
        :param data: 原始数据字典，格式为 {股票代码: DataFrame}
        :param metrics: 需要聚合的指标列表，None表示聚合全部指标
        :return 按指标聚合的字典，格式为 {指标名: DataFrame}
        """
        if not data:
            return {}

        # 获取所有指标名（使用第一个股票的列作为默认值）
        all_metrics = data[next(iter(data))].columns.tolist()
        target_metrics = metrics if metrics is not None else all_metrics

        panel_data = {}
        for metric in target_metrics:
            # 收集所有包含该指标的股票数据
            stock_series = {
                code: df[metric]
                for code, df in data.items()
                if metric in df.columns
            }
            if stock_series:
                metric_df = pd.DataFrame.from_dict(stock_series)
                panel_data[metric] = metric_df

        return panel_data

    @classmethod
    def _get_valid_data(
            cls,
            data: dict[str, pd.DataFrame],
            segment: str
    ) -> dict[str, pd.DataFrame]:
        """筛选有用数据"""
        return {
            metric: df.iloc[0, :].to_frame(name="slope")
            for metric, df in data.items()
            if segment in metric
        }

    def _reset_index(
            self,
            data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """重置列名，代码 -> 企业简称"""
        # 企业简称映射字典
        short_name_mapping: dict = (
            self.industry_codes
            .set_index("股票代码")
            ["公司简称"]
            .to_dict()
        )
        return {
            date: df_.rename(index=short_name_mapping)
            for date, df_ in data.items()
        }

    def _add_industry(
            self,
            data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        加入行业分类数据
        :param data: 原始数据
        :return: 具有行业信息的数据
        """
        industry = (
            self.industry_codes[["股票代码", "一级行业", "二级行业", "三级行业"]]
            .set_index("股票代码")
        )
        return {
            date: df.join(industry, how="left")
            .fillna(
                {
                    "一级行业": "未知行业",
                    "二级行业": "未知行业",
                    "三级行业": "未知行业",
                }
            )
            for date, df in data.items()
        }

    # --------------------------
    # 行业统计方法
    # --------------------------
    def _stats_industry(
            self,
            critical_value: int
    ) -> dict[str, pd.DataFrame]:
        """
        统计行业信息
        :param critical_value: 临界值
        """
        result = {}

        for metric, df in self.kline.items():
            # 斜率筛选出 >= 5
            filtered_df = df[df["slope"] >= critical_value]

            result[f"{metric}_industry_count"] = pd.concat(
                [
                    self._calc_counts_and_percent(filtered_df, df, "一级行业"),
                    self._calc_counts_and_percent(filtered_df, df, "二级行业"),
                    self._calc_counts_and_percent(filtered_df, df, "三级行业"),
                ],
                axis=1
            )

        return result

    @staticmethod
    def _calc_counts_and_percent(
            filtered_df: pd.DataFrame,
            df: pd.DataFrame,
            industry_col: str
    ) -> pd.DataFrame:
        """
        计算行业数与占自身的比重
        :param filtered_df: 过滤后的数据
        :param df: 原始数据
        :param industry_col: 行业名
        :return: 行业数及其占比
        """
        industry_counts = df[industry_col].value_counts()
        filtered_counts = filtered_df.value_counts(industry_col).rename(industry_col)
        filtered_percent = (
            (filtered_counts / industry_counts * 100)
            .rename(f"{industry_col}%")
            .dropna()
        )
        return pd.concat(
            [
                filtered_counts,
                filtered_percent,
            ],
            axis=1
        )

    # --------------------------
    # 存储、可视化方法
    # --------------------------
    def _store_results(
            self
    ) -> None:
        """存储分析结果"""
        storage = DataStorage(self.result_data_path)
        storage.write_dict_to_parquet(
            self.kline,
            merge_original_data=False
        )

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(
            self
    ) -> None:
        """分析流程"""
        try:
            # -1 加载数据
            self._load_data()
            # -2 整合数据
            self._aggregate_into_panel_data()
            # -3 统计行业
            self.kline.update(
                self._stats_industry(self.slope_critical)
            )
            # -4 存储
            self._store_results()
        except Exception as e:
            self.logger.error(f"分析流程异常终止: {str(e)}") if self.logger else print(f"Error: {str(e)}")
            raise
