"""
分析业务层（个体、中观、宏观、量化）
"""

import pandas as pd
from loguru import logger
from pathlib import Path
from pathos.multiprocessing import ProcessPool as Pool

from constant.path_config import DataPATH
from constant.type_ import (
    CYCLE, FINANCIAL_CYCLE, CLASS_LEVEL, DIMENSION,
    INDUSTRY_SHEET, KLINE_SHEET, WEIGHTS,
    validate_literal_params
)
from utils.data_processor import DataProcessor
from data_storage import DataStorage
from data_loader import DataLoader
from utils.drawer import Drawer, IndividualDrawer
from stats_metrics import StatisticsMetrics, IndividualStatisticsMetrics
from kline_metrics import KLineMetrics
from valuation_metrics import ValuationMetrics
from financial_metrics import FinancialMetrics
from governance_metrics import GovernanceMetrics

import warnings
warnings.filterwarnings(
    "ignore",
    category=pd.errors.PerformanceWarning
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

CLEANED_METRICS = [
    # 资产负债表
    # 资产类
    '货币资金占比',
    '流动资产占比',
    '经营性资产占比',
    '金融性资产占比',

    # 负债类
    '经营性负债占比',
    '金融性负债占比',

    # 股东权益类
    '股东入资占比',
    '利润留存占比',

    # 利润表
    '营业税金负担率',
    '核心利润占比',

    '销售费用率',
    '管理费用率',
    '财务费用率',
    '研发费用率',
    '期间费用率',
    '毛销差',

    '减值损失占比',
    '资产减值损失占比',
    '信用减值损失占比',

    '所得税税率',
    '平均所得税税率',
    '股利支付率',
    '利润留存率',

    '内含增长率',
    '可持续增长率',

    # 比率指标
    # 偿债类
    '流动比率',
    '速动比率',
    '现金比率',
    '产权比率',
    '权益乘数',
    '净经营资产权益乘数',
    '净财务杠杆',
    '现金流量比率',
    '长期资本化比率',
    '债务资本化比率',
    '资产负债率',
    '有息负债率',
    '利息保障倍数',
    '现金流量利息保障倍数',

    # 盈利类
    '经营性资产收益率',
    '金融性资产收益率',
    '净经营资产净利率',
    '营业净利率',
    '经营净利率',
    '权益净利率',
    '归母权益净利率',
    '总资产净利率',

    '核心利润获现率',
    '核心利润净利率',
    '收现比',
    '净现比',

    '税后利息率',
    '经营差异率',
    '杠杆贡献率',

    # 营运类
    '存货周转天数',
    '应收票据及应收账款周转天数',
    '应付票据及应付账款周转天数',
    '固定资产周转天数',
    '流动资产周转天数',
    '非流动资产周转天数',
    '总资产周转天数',
    '营业周期',
    '现金转换周期',

    '固定资产周转率',
    '流动资产周转率',
    '非流动资产周转率',
    '营运资本周转率',
    '净经营资产周转率',
    '总资产周转率',

    '单位营收所需的经营性营运资本',

    '应付款比率',
    '预收款比率',
    '上下游资金占用',

    # 成长类
    '固定资产原值推动力',
    '固定资产净值推动力',
    '扩张倍数',
    '收缩倍数',

    # 估值
    '市盈率',
    '核心利润市盈率',
    '盈利市值比',
    '核心利润盈利市值比',
    '市净率',
    '账面市值比',
    '市销率',
    '市盈率(平均ROE)',
    '股息率',
    'PEG',
    '实际收益率',
    '归母实际收益率',
    '核心利润实际收益率',
]


###############################################################
class Analyzer:

    @validate_literal_params
    def __init__(
            self,
            dimension: DIMENSION,
            cycle: CYCLE,
            financial_cycle: FINANCIAL_CYCLE,
            params,
            start_date: str,
            end_date: str,
            financial_end_date: str,
            storage_dir_name: str,
            target_info: dict[str, str] | list[str] | str,
            index_code: str,
            code_range: INDUSTRY_SHEET = "A",
            weight_name: WEIGHTS = "等权",
            kline_adjust: KLINE_SHEET = "backward_adjusted",
            class_level: CLASS_LEVEL = "一级行业",
            draw_filter: bool = False,
            quant: bool = False,
            processes_nums: int = 1,
            trade_status_filter: bool = False,
            debug: bool = False
    ):
        """
        :param dimension: 维度
        :param cycle: 周期
        :param financial_cycle: 财务周期
        :param params: 指定参数
        :param start_date: 起始时间
        :param end_date: 结束时间
        :param financial_end_date: 财务数据结束时间
        :param storage_dir_name: 存储文件夹名
        :param target_info: 行业信息 -1 {"白酒": 3} -2 ["688305", "601882"] -3 "601882"
        :param quant: 是否为量化分析
        :param weight_name: 加权名
        :param kline_adjust: k线复权方法
        :param class_level: 行业级别
        :param draw_filter: 绘图过滤
        :param index_code: 指数代码
        :param code_range: 代码范围
        :param processes_nums: 多进程数
        :param trade_status_filter: 交易状态过滤
        :param debug: debug模式
        """
        self.dimension = dimension
        self.PARAMS = params
        self.cycle = cycle
        self.financial_cycle = financial_cycle
        self.start_date = start_date
        self.end_date = end_date
        self.financial_end_date = financial_end_date
        self.storage_file_name = storage_dir_name
        self.target_info = target_info
        self.weight_name = weight_name
        self.kline_adjust = kline_adjust
        self.class_level = class_level
        self.draw_filter = draw_filter
        self.index_code = index_code
        self.code_range = code_range
        self.quant = quant
        self.processes_nums = processes_nums
        self.CLEANED_METRICS = CLEANED_METRICS
        self.trade_status_filter = trade_status_filter
        self.debug = debug

        # --------------------------
        # 初始化配置参数
        # --------------------------
        # 工具类
        self.loader = DataLoader
        self.process = DataProcessor

        # 数据容器
        self.data_container = self._setup_data_container()

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
        # 设置模式
        # --------------------------
        self.mode = self._setup_mode()

        # --------------------------
        # 路径参数
        # --------------------------
        # 数据路径
        self.financial_data_path = DataPATH.FINANCIAL_DATA
        self.kline_stock_path = DataPATH.STOCK_KLINE_DATA

        # 存储路径
        self.result_data_path = self._setup_result_path()

        # --------------------------
        # 其他参数
        # --------------------------
        # 目标代码
        self.target_codes = self._get_target_codes()
        # 目标行业字典
        self.industry_mapping = self._get_industry_codes()
        # 所需数据代码
        self.load_data_codes = self._get_load_data_codes()
        # 派生参数
        self._derive_parameters()
        # 指数k线
        self.index_kline = None

        # --------------------------
        # 映射参数
        # --------------------------
        # 方法映射
        self.MAPPING = self._load_function()

    @classmethod
    def _setup_data_container(
            cls
    ) -> dict[str, dict]:
        """设置数据存储容器"""
        return {
            "financial": {},            # 财务
            "rolling_financial": {},    # 滚动财务
            "governance": {},           # 公司治理
            "kline": {},                # 量价
            "valuation": {},            # 估值
            "stat": {},                 # 统计
            "weight": {}                # 加权权重
        }

    def _setup_mode(
            self
    ) -> str:
        """设置模式"""
        return (
            "individual" if isinstance(self.target_info, str)
            else "quant" if self.quant else "normal"
        )

    def _setup_result_path(
            self
    ) -> Path:
        """设置结果存储路径"""
        result_path = {
            "mac": DataPATH.MAC_ANALYSIS_RESULT,
            "meso": DataPATH.MESO_ANALYSIS_RESULT,
            "micro": DataPATH.MICRO_ANALYSIS_RESULT,
            "individual": DataPATH.INDIVIDUAL_ANALYSIS_RESULT,
            "quant": DataPATH.QUANT_ANALYSIS_RESULT
        }
        return (
            result_path.get(self.mode) / self.storage_file_name if self.mode in ["individual", "quant"]
            else result_path.get(self.dimension) / f"{self.storage_file_name}-{self.weight_name}"
        )

    def _derive_parameters(
            self
    ) -> None:
        """计算派生参数"""
        # --------------------------
        # k线参数
        # --------------------------
        # k线起始时间 比财务起始时间早一年
        self.kline_start_date = pd.to_datetime(self.start_date) - pd.DateOffset(years=1)
        self.kline_start_date = self.kline_start_date.strftime("%Y-%m-%d")
        # 时间月末对齐 仅适用于周度数据之上
        self.aligned_to_month_end = False if self.cycle in ["week", "day"] else True

        # --------------------------
        # 财务参数
        # --------------------------
        # 财务数据填充日期
        self.financial_fill_date = (
            None if self.cycle in ["quarter", "half", "year"]
            else self.loader.get_trading_calendar(
                self.cycle,
                self.start_date,
                self.end_date,
                self.aligned_to_month_end
            )
        )

        # 是否检查k线长度
        self.financial_inspection = False if self.dimension == "micro" else True

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
        # -1 个股
        if self.mode == "individual":
            if isinstance(self.target_info, str):
                return [self.target_info]
            else:
                raise TypeError(f"个股分析模式不支持该数据类型 {type(self.target_info)}")

        # -2 一般/量化
        # -2-1 宏观 全部代码
        if self.dimension == "mac":
            return self.loader.get_industry_codes(
                sheet_name=self.code_range,
                industry_info={"全部": "一级行业"},
                return_type="list"
            )
        # -2-2 中观/微观 字典
        if isinstance(self.target_info, dict):
            return self.loader.get_industry_codes(
                sheet_name=self.code_range,
                industry_info=self.target_info,
                return_type="list"
            )
        # -2-3 中观/微观 列表
        elif isinstance(self.target_info, list):
            return self.target_info
        else:
            raise TypeError(f"一般分析/量化分析不支持该数据类型 {type(self.target_info)}")

    def _get_industry_codes(
            self
    ) -> pd.DataFrame:
        """获取目标行业字典"""
        # -1 中观/字典 特定行业字典
        if self.dimension == "meso" and isinstance(self.target_info, dict):
            return self.loader.get_industry_codes(
                    sheet_name=self.code_range,
                    industry_info=self.target_info,
                ).assign(**{"股票代码": lambda df: df["股票代码"].str.split(".").str[0]})

        # -2 其他
        else:
            return self.loader.get_industry_codes(
                sheet_name=self.code_range,
                industry_info={"全部": ""},
            ).assign(**{"股票代码": lambda df: df["股票代码"].str.split(".").str[0]})

    def _get_load_data_codes(
            self
    ) -> list[str]:
        """获取所需数据代码"""
        return self.target_codes

    def _load_function(
            self
    ) -> dict:
        """获取指标方法名（财务、估值、k线、统计）"""
        return {
            "FINANCIAL": self.loader.load_yaml_file(
                DataPATH.FINANCIAL_METRICS,
                "FUNCTION_OF_FINANCIAL"
            ),
            "KLINE": self.loader.load_yaml_file(
                DataPATH.KLINE_METRICS,
                "FUNCTION_OF_KLINE"
            ),
            "VALUATION": self.loader.load_yaml_file(
                DataPATH.VALUATION_METRICS,
                "FUNCTION_OF_VALUATION"
            ),
            "GOVERNANCE": self.loader.load_yaml_file(
                DataPATH.GOVERNANCE_METRICS,
                "FUNCTION_OF_GOVERNANCE"
            ),
            "STATISTICS": self.loader.load_yaml_file(
                DataPATH.STATISTICS_METRICS,
                "FUNCTION_OF_STATISTICS"
            ),
        }

    # --------------------------
    # 数据处理流程
    # --------------------------
    def _load_data(
            self
    ) -> None:
        """加载数据"""
        # 使用多进程池并行处理
        with Pool(self.processes_nums) as pool:
            results = pool.map(self._process_single_company, self.load_data_codes)

        # 合并处理结果到数据容器
        self._unpack_results(results)

    def _load_data_debug(
            self
    ) -> None:
        """加载数据"""
        results = []
        for code in self.load_data_codes[: 20]:
            self.logger.info(f"{code}: 加载/计算指标")
            results.append(
                self._process_single_company_debug(code)
            )

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
            self.data_container["rolling_financial"][code] = result["rolling_financial"]
            if not result["financial"].empty:
                self.data_container["financial"][code] = result["financial"]
            self.data_container["valuation"][code] = result["valuation"]
            self.data_container["kline"][code] = result["kline"]
            self.data_container["governance"][code] = result["governance"]
            self.data_container["weight"][code] = result["weight"]

    def _process_single_company_debug(
            self,
            code: str
    ) -> dict[str, str | pd.DataFrame]:
        """处理单个企业数据"""
        result = {"code": code}
        # --------------------------
        # 加载数据
        # --------------------------
        # -1 估值使用后复权数据 -2 技术使用前复权数据
        backward_adjusted_kline = self._load_kline_data(code, "backward_adjusted")
        split_adjusted_kline = self._load_kline_data(code, "split_adjusted")
        
        bonus = self._load_bonus(code)

        # --------------------------
        # 指标计算（技术指标/基本面指标/公司治理指标）
        # --------------------------
        split_adjusted_kline = self._calculate_kline(
            split_adjusted_kline,
            self.index_kline
        )
        backward_adjusted_kline = self._calculate_kline(
            backward_adjusted_kline,
            self.index_kline
        )

        financial = self._get_financial_data(
            code,
            self.financial_cycle,
            bonus
        )

        rolling_financial = self._get_financial_data(
            code,
            self.financial_cycle,
            bonus,
            True
        )
        rolling_financial_to_value = (
            rolling_financial.copy(deep=True) if self.financial_cycle == "quarter"
            else self._get_financial_data(
                code,
                "quarter",
                bonus,
                True
            )
        )

        governance = self._calc_governance(code)

        # --------------------------
        # 时间索引转变为披露时间、填充数据(财务/公司治理）
        # --------------------------
        if self.mode == "quant":
            rolling_financial.index = self._convert_to_disclose_date(rolling_financial.index)
            rolling_financial_to_value.index = self._convert_to_disclose_date(rolling_financial_to_value.index)
            rolling_financial = self._fill_financial_data(rolling_financial)

            if not governance.empty:
                governance.index = self._convert_to_disclose_date(governance.index)
                governance = self._fill_financial_data(governance)

        # 任意模式，均需填充
        rolling_financial_to_value = self._fill_financial_data(rolling_financial_to_value)

        # --------------------------
        # 估值指标计算
        # --------------------------
        # 加载总股本
        shares = self._load_shares(
            code,
            financial_date=rolling_financial_to_value.index
        )
        # 估值
        value = self._calculate_valuation(
            rolling_financial_to_value,
            backward_adjusted_kline,
            bonus,
            shares
        )

        # --------------------------
        # 权重计算
        # --------------------------
        if self.dimension == "micro" or self.weight_name == "等权":
            weight = pd.DataFrame()
        else:
            weight = (
                value["市值"] if self.weight_name == "市值"
                else rolling_financial_to_value[self.weight_name]
            ).to_frame(self.weight_name)

        # --------------------------
        # 指标存储 （量化无财务数据，仅有滚动财务数据），此外，仅量化完成财务数据填充
        # --------------------------
        result.update({
            "rolling_financial": rolling_financial,
            "financial": financial if self.mode != "quant" else pd.DataFrame(),
            "valuation": value,
            "governance": governance,
            "kline": split_adjusted_kline,
            "weight": weight
        })

        return result

    def _process_single_company(
            self,
            code: str
    ) -> dict[str, str | pd.DataFrame]:
        """处理单个企业数据"""
        result = {"code": code}
        try:
            # --------------------------
            # 加载数据
            # --------------------------
            backward_adjusted_kline = self._load_kline_data(code, "backward_adjusted")
            split_adjusted_kline = self._load_kline_data(code, "split_adjusted")
            bonus = self._load_bonus(code)

            # --------------------------
            # 指标计算（技术指标/基本面指标/公司治理指标）
            # --------------------------
            split_adjusted_kline = self._calculate_kline(
                split_adjusted_kline,
                self.index_kline
            )
            backward_adjusted_kline = self._calculate_kline(
                backward_adjusted_kline,
                self.index_kline
            )

            financial = self._get_financial_data(
                code,
                self.financial_cycle,
                bonus
            )
            rolling_financial = self._get_financial_data(
                code,
                self.financial_cycle,
                bonus,
                True
            )
            rolling_financial_to_value = (
                rolling_financial.copy(deep=True) if self.financial_cycle == "quarter"
                else self._get_financial_data(
                    code,
                    "quarter",
                    bonus,
                    True
                )
            )

            governance = self._calc_governance(code)

            # --------------------------
            # 时间索引转变为披露时间、填充数据(财务/公司治理）
            # --------------------------
            if self.mode == "quant":
                rolling_financial.index = self._convert_to_disclose_date(rolling_financial.index)
                rolling_financial_to_value.index = self._convert_to_disclose_date(rolling_financial_to_value.index)
                rolling_financial = self._fill_financial_data(rolling_financial)

                if not governance.empty:
                    governance.index = self._convert_to_disclose_date(governance.index)
                    governance = self._fill_financial_data(governance)

            # 任意模式，均需填充
            rolling_financial_to_value = self._fill_financial_data(rolling_financial_to_value)

            # --------------------------
            # 估值指标计算
            # --------------------------
            # 加载总股本
            shares = self._load_shares(
                code,
                financial_date=rolling_financial_to_value.index
            )
            # 估值
            value = self._calculate_valuation(
                rolling_financial_to_value,
                backward_adjusted_kline,
                bonus,
                shares
            )

            # --------------------------
            # 权重计算
            # --------------------------
            if self.dimension == "micro" or self.weight_name == "等权":
                weight = pd.DataFrame()
            else:
                weight = (
                    value["市值"] if self.weight_name == "市值"
                    else rolling_financial_to_value[self.weight_name]
                ).to_frame(self.weight_name)

            # --------------------------
            # 指标存储 （量化无财务数据，仅有滚动财务数据），此外，仅量化完成财务数据填充
            # --------------------------
            result.update({
                "rolling_financial": rolling_financial,
                "financial": financial if self.mode != "quant" else pd.DataFrame(),
                "valuation": value,
                "governance": governance,
                "kline": split_adjusted_kline,
                "weight": weight
            })
        except Exception as e:
            result.update({
                "error": f"{code} - {str(e)}"
            })

        return result

    # --------------------------
    # 数据加载方法
    # --------------------------
    def _load_index_data(
            self
    ) -> None:
        """加载指数数据"""
        self.index_kline =  self.loader.get_index_kline(
            code=self.index_code,
            cycle=self.cycle,
            start_date=self.kline_start_date,
            end_date=self.end_date,
            aligned_to_month_end=self.aligned_to_month_end
        )

    def _load_kline_data(
            self,
            code: str,
            kline_adjust: KLINE_SHEET,
    ) -> pd.DataFrame:
        """
        加载k线数据
        :param code: 代码
        :param kline_adjust: k线加权
        """
        df = self.loader.get_kline(
            code=code,
            cycle=self.cycle,
            adjusted_mode=kline_adjust,
            start_date=self.kline_start_date,
            end_date=self.end_date,
            aligned_to_month_end=self.aligned_to_month_end
        )

        # 交易状态过滤
        if self.trade_status_filter:
            df = df[df["tradestatus"] == 1]

        # 交易日期过滤（部分退市股出现截止日期不统一的问题）
        if self.financial_fill_date is not None:
            df = df[df.index.isin(self.financial_fill_date)]

        return df.drop("code", axis=1)

    def _get_financial_data(
            self,
            code: str,
            cycle: FINANCIAL_CYCLE,
            bonus: pd.DataFrame,
            rolling: bool = False
    ) -> pd.DataFrame:
        """
        加载财务数据
        :param code: 代码
        :param cycle: 周期
        :param rolling: 是否为滚动值
        """
        func = self.loader.get_rolling_financial if rolling else self.loader.get_financial
        df = func(
            code=code,
            cycle=cycle,
            start_date=self.start_date,
            end_date=self.financial_end_date,
            inspection=self.financial_inspection
        )
        calculator = FinancialMetrics(
            financial_data=df,
            bonus_data=bonus,
            cycle=cycle,
            methods=sum(self.PARAMS.finance.financial_analysis.values(), []),
            function_map=self.MAPPING["FINANCIAL"],
            de_extreme_method=self.process.percentile
        )
        calculator.calculate()

        return calculator.metrics.round(4)

    def _load_bonus(
            self,
            code: str
    ) -> pd.DataFrame:
        """
        加载分红数据
        :param code: 企业代码
        """
        return self.loader.get_bonus(code=code)
        
    def _load_shares(
            self,
            code: str,
            financial_date: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        加载总股本数据
        :param code: 企业代码
        :param financial_date:
        """
        return self.loader.get_total_shares(
            code=code,
            financial_date=financial_date
        )

    # --------------------------
    # 填充（财务）
    # 计算（财务、估值、量价）
    # --------------------------
    @classmethod
    def _convert_to_disclose_date(
            cls,
            date_index: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """
        将财报时间转变为财务披露时间
            -1 12 -> 4
            -2 3 -> 4
            -3 6 -> 8
            -4 9 -> 10
        """
        def __convert_to_disclose_date(date):
            if date.month == 3:
                return date.replace(month=4, day=1) + pd.tseries.offsets.MonthEnd(0)
            elif date.month == 6:
                return date.replace(month=8, day=1) + pd.tseries.offsets.MonthEnd(0)
            elif date.month == 9:
                return date.replace(month=10, day=1) + pd.tseries.offsets.MonthEnd(0)
            elif date.month == 12:
                return date.replace(month=4, year=date.year + 1, day=1) + pd.tseries.offsets.MonthEnd(0)
            else:
                raise ValueError("财报日期不属于 3/6/9/12 月")

        return date_index.map(__convert_to_disclose_date)

    def _fill_financial_data(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """财务数据填充方法，支持日/周/月数据"""
        return df.groupby(level=0).last().reindex(self.financial_fill_date, method="ffill").dropna(how="all")

    def _calculate_valuation(
            self,
            financial: pd.DataFrame,
            kline: pd.DataFrame,
            bonus: pd.DataFrame,
            shares: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算估值指标"""
        calculator = ValuationMetrics(
            financial_data=financial,
            kline_data=kline,
            bonus_data=bonus,
            shares_data=shares,
            cycle=self.cycle,
            methods=sum(self.PARAMS.valuation.__dict__.values(), []),
            function_map=self.MAPPING["VALUATION"],
            kline_adjust="backward_adjusted",
        )
        calculator.calculate()

        return calculator.metrics.round(4)

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
            function_map=self.MAPPING["KLINE"]
        )
        calculator.calculate()
        return calculator.metrics.round(4)

    def _calc_governance(
            self,
            code: str
    ) -> pd.DataFrame:
        """
        加载股东数据
        :param code: 企业代码
        """
        governance = GovernanceMetrics(
            shareholders=self.loader.get_top_ten_shareholders(code),
            # circulating_shareholders=self.loader.get_top_ten_circulating_shareholders(code),
            methods=sum(self.PARAMS.governance.__dict__.values(), []),
            function_map=self.MAPPING["GOVERNANCE"],
        )
        governance.calculate()
        return governance.metrics.round(2)

    # --------------------------
    # 聚合、合成、统计方法
    # --------------------------
    def _aggregate_into_panel_data(
            self
    ) -> dict:
        """处理数据聚合"""
        aggregate = {
            "financial": sum(self.PARAMS.finance.financial_analysis.values(), [])
            + sum(self.PARAMS.finance.basic_reports.values(), []),
            "rolling_financial": sum(self.PARAMS.finance.financial_analysis.values(), [])
            + sum(self.PARAMS.finance.basic_reports.values(), []),
            "kline": None,
            "valuation": sum(self.PARAMS.valuation.__dict__.values(), []),
            "governance": sum(self.PARAMS.governance.__dict__.values(), []),
            "weight": None,
        }
        return {
            data_name: self.__aggregate_into_panel_data(
                self.data_container[data_name],
                agg_metrics
            ) for data_name, agg_metrics in aggregate.items()
        }

    def _clean_data(
        self,
        raw_data: dict,
        cleaning_func: callable
    ) -> dict:
        """
        清洗数据
        :param raw_data: 原始数据
        :param cleaning_func: 清洗方法
        :return: 清洗后的数据
        """
        categories = [
            "financial", "rolling_financial", "valuation", "weight"
        ]
        # 转置->清洗->转置还原
        return {
            category: {
                metric: (
                    cleaning_func(df.T).T
                    if metric in self.CLEANED_METRICS
                    else df
                )
                for metric, df in raw_data.get(category, {}).items()
            }
            for category in categories
        }

    def _synthesis_into_mac_or_industry(
            self,
            micro_data: dict,
    ) -> dict:
        """
        数据清洗和加权
        :param micro_data: 个股截面数据
        :return: 宏观/行业数据
        """
        weights = micro_data["weight"][self.weight_name]
        weight_categories = {
            "financial": weights,
            "rolling_financial": weights,
            "valuation": weights,
            "kline": pd.DataFrame()
        }
        return {
            category: {
                metric: self.__synthesis_into_mac_or_industry(
                    df,
                    self.industry_mapping,
                    self.class_level,
                    self.dimension,
                    weight
                )
                for metric, df in micro_data.get(category, {}).items()
            }
            for category, weight in weight_categories.items()
        }

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

    @staticmethod
    def __synthesis_into_mac_or_industry(
        data: pd.DataFrame,
        industry: pd.DataFrame,
        class_level: str,
        dimension: str,
        weights: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算等权平均值
        :param data: 按指标聚合的数据
        :param industry: 目标行业字典
        :param class_level: 行业级数
        :param dimension: 维度 "mac", "meso"
        :param weights: 权重数据 -1 None -> 等权 -2 非None -> 加权
        :return 按行业合成的DataFrame
        """
        # -1 mac合成
        if dimension == "mac":
            return data.mean(axis=1).to_frame(name="mac").round(4)

        # -2 行业合成
        industry_means = {}
        for industry, df in industry.groupby(class_level):
            industry_codes = df["股票代码"].tolist()
            valid_codes = list(set(data.columns) & set(industry_codes))
            if valid_codes:
                if weights.empty:
                    industry_means[industry] = data[valid_codes].mean(axis=1, numeric_only=True)
                else:
                    # 有效行业数据
                    industry_valid = data[valid_codes]
                    # 合成权重
                    industry_weights = weights.loc[industry_valid.index, valid_codes]
                    norm_weights = industry_weights.div(
                        industry_weights.sum(axis=1),
                        axis=0
                    )
                    # 加权计算
                    industry_means[industry] = (industry_valid * norm_weights).sum(axis=1)

        return pd.DataFrame.from_dict(industry_means).round(4) if industry_means else pd.DataFrame()

    def _summarize_data(
            self
    ) -> dict:
        """数据汇总处理"""
        if self.mode == "individual":
            return self.data_container

        # -1 聚合为面板数据
        panel_data = self._aggregate_into_panel_data()
        if self.dimension == "micro":
            return panel_data

        # -2 数据清洗（去极值）
        cleand_data = self._clean_data(
            raw_data=panel_data,
            cleaning_func=self.process.percentile
        )

        # -3 合成为 mac|行业
        synthesis_data = self._synthesis_into_mac_or_industry(
            micro_data=cleand_data,
        )

        return synthesis_data

    # --------------------------
    # 统计、存储、可视化方法
    # --------------------------
    def _perform_statistics(
            self
    ) -> dict:
        """执行统计分析"""
        if self.mode == "individual":
            stat = IndividualStatisticsMetrics(
                code=self.target_info,
                data_container=self.data_container,
                cycle=self.cycle,
                financial_cycle=self.financial_cycle,
                methods=sum(self.PARAMS.stat.__dict__.values(), []),
                function_map=self.MAPPING["STATISTICS"],
                index_kline=self.index_kline
            )
        else:
            stat = StatisticsMetrics(
                data_container=self.data_container,
                cycle=self.cycle,
                financial_cycle=self.cycle if self.mode == "quant" else self.financial_cycle,
                methods=sum(self.PARAMS.stat.__dict__.values(), []),
                function_map=self.MAPPING["STATISTICS"],
                index_kline=self.index_kline
            )
        stat.calculate()
        return stat.data_container

    def _reset_columns(
            self
    ) -> None:
        """重置列名，代码 -> 企业简称"""
        if self.mode == "normal" and self.dimension == "micro":
            # 企业简称映射字典
            short_name_mapping: dict = (
                self.industry_mapping
                .set_index("股票代码")
                ["公司简称"]
                .to_dict()
            )

            for data in self.data_container.values():
                for df_ in data.values():
                    df_.rename(
                        short_name_mapping,
                        inplace=True,
                        axis=1 if isinstance(df_, pd.DataFrame) else 0
                    )

    def _save_data(
            self,
            df: pd.DataFrame,
            file_name: str,
    ) -> None:
        """
        存储数据（直接覆盖）
        :param df: 数据
        :param file_name: 文件名
        """
        DataStorage(self.result_data_path).write_df_to_parquet(
            df,
            file_name,
            index=False,
            merge_original_data=False,
            subset=["date"] if "date" in df.columns else None,
            sort_by=["date"] if "date" in df.columns else None,
        )

    def _store_results(
            self
    ) -> None:
        """存储分析结果"""
        if self.mode == "individual":
            return
        else:
            for container, data in self.data_container.items():
                if data:
                    for metric, df in data.items():
                        if isinstance(df, pd.Series):
                            df = pd.DataFrame({metric: df})
                        # 非quant模式，滚动财务数据需加上前缀，以免混淆
                        if self.mode != "quant" and container == "rolling_financial":
                            metric = f"rolling_{metric}"
                        self._save_data(df.reset_index(), metric)

    def _draw_charts(
            self
    ) -> None:
        """生成可视化图表"""
        if self.mode == "individual":
            self.drawer = IndividualDrawer(
                path=self.result_data_path,
                pages_name=self.PARAMS.visualization.pages_name,
                pages_config=self.PARAMS.visualization.pages_config,
                data_container=self.data_container
            )
        else:
            self.drawer = Drawer(
                path=self.result_data_path,
                pages_name=self.PARAMS.visualization.pages_name,
                pages_config=self.PARAMS.visualization.pages_config,
                data_container=self.data_container
            )
        self.drawer.draw()
        self.drawer.render()

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(
            self
    ) -> None:
        """分析流程"""
        self.logger.info(f"分析流程开始")

        # -1 加载数据
        self._load_index_data()

        if self.debug:
            self._load_data_debug()
        else:
            self._load_data()

        # -2 整合数据
        self.data_container = self._summarize_data()

        # -3 面板统计
        self.data_container = self._perform_statistics()

        # -4 列名转换
        self._reset_columns()

        # -5 存储
        self._store_results()

        # -6 可视化
        if self.draw_filter:
            self._draw_charts()

        self.logger.info(f"分析流程结束")
