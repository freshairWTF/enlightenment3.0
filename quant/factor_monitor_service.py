"""
因子监控业务层
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Literal, get_args
from collections import defaultdict

import pandas as pd

from range_filter import RangeFilter
from base_service import BaseService
from constant.type_ import CYCLE, CLASS_LEVEL, FILTER_MODE, validate_literal_params
from constant.monitor_setting import Factor
from constant.path_config import DataPATH


#####################################################
class FactorMonitor(BaseService):
    """因子监控"""

    CORE_FACTOR = [
        "对数市值", "close", "pctChg", '核心利润盈利市值比', '市净率', "市销率", "turn"
    ]
    VALUE_SPREAD_PARAMETERS = ['核心利润盈利市值比', '市净率', "市销率"]

    @validate_literal_params
    def __init__(
            self,
            source_dir: str | Path,
            storage_dir: str | Path,
            factors_setting: list[dataclass],
            cycle: CYCLE,
            class_level: CLASS_LEVEL = "1",
            lag_period: int = 1,
            benchmark_code: str = "000300",
            group_nums: int = 5
    ):
        """
        :param source_dir: 原始数据目录名称
        :param storage_dir: 存储数据目录名称
        :param factors_setting: 因子设置
        :param cycle: 周期
        :param class_level: 行业级数
        :param lag_period: 滞后期数
        :param benchmark_code: 基准指数代码
        :param group_nums: 分组数
        """
        self.storage_dir = storage_dir
        self.source_dir = source_dir
        self.factors_setting = factors_setting
        self.cycle = cycle
        self.class_level = class_level
        self.lag_period = lag_period
        self.benchmark_code = benchmark_code
        self.group_nums = group_nums

        # --------------------------
        # 初始化配置参数
        # --------------------------
        self.filter = RangeFilter
        # 初始化其他配置
        self._initialize_config()
        # 日志
        self.logger = self.setup_logger(self.storage_dir)

    # --------------------------
    # 初始化
    # --------------------------
    def _initialize_config(
            self
    ) -> None:
        """初始化配置参数"""
        # --------------------------
        # 路径参数
        # --------------------------
        self.storage_dir = DataPATH.QUANT_FACTOR_MONITOR_RESULT / self.storage_dir
        self.source_dir = DataPATH.QUANT_CONVERT_RESULT / self.source_dir
        
        # --------------------------
        # 数据
        # --------------------------
        self.industry_mapping = self.load_industry_mapping()
        self.listed_nums = self.load_listed_nums()
        self.raw_data = self.load_factor_data(self.source_dir)

        # --------------------------
        # 因子名
        # --------------------------
        self.support_factors_name = self._get_support_factors()
        self.factors_name = self._get_factors_name()
        self.processed_factors_name = self._get_processed_factors_name()
        self.valid_factors_name = self._get_valid_factor()

        # --------------------------
        # 其他参数
        # --------------------------
        self.visual_setting = Factor().visualization
        self.group_label = self.setup_group_label(self.group_nums)
        self.recent_period = 4
        self.filter_mode = Literal[
                    "_entire_filter",
                    "_overall_filter",
                    "_large_cap_filter",
                    "_mega_cap_filter",
                    "_small_cap_filter"
                ]
        
    def _get_support_factors(
            self
    ) -> list[str]:
        """获取分析因子之外所需的因子"""
        factors = (
            self.filter.PARAMETERS
            + self.CORE_FACTOR
            + self.VALUE_SPREAD_PARAMETERS
        )
        return [f for f in set(factors) if f]

    def _get_factors_name(
            self
    ) -> list[str]:
        """获取因子名"""
        return [
            setting.factor_name
            for setting in self.factors_setting
        ]

    def _get_denominators_name(
            self
    ) -> list[str]:
        """获取重构因子所需的分母"""
        return [
            setting.restructure_denominator
            for setting in self.factors_setting
            if setting.restructure
        ]

    def _get_processed_factors_name(
            self
    ) -> list[str]:
        """获取预处理因子名"""
        return [
            f"processed_{setting.factor_name}"
            for setting in self.factors_setting
        ]

    def _get_valid_factor(
            self
    ) -> list:
        """获取全部所需的因子"""
        return list(set(
            self.factors_name
            + self.support_factors_name
        ))

    # --------------------------
    # 单范围初始化
    # --------------------------
    def _get_storage_dir(
            self,
            filter_mode: str 
    ) -> Path:
        """获取存储地址"""
        return (
            self.storage_dir /
            f"股票池{filter_mode}-滞后{self.lag_period}{self.cycle[0]}"
        )

    # --------------------------
    # 数据处理
    # --------------------------
    def _data_process(
            self,
            raw_data: dict[str, pd.DataFrame],
            valid_factors: list[str],
            filter_mode: FILTER_MODE,
            factors_setting: list[dataclass]
    ) -> dict[str, pd.DataFrame]:
        """
        数据处理
            -1 截取（全部因子） -2 平移（除 pctChg 之外） -3 过滤（过滤因子）
            -4 预处理（回测因子） -5 分组（预处理因子、使用过去的因子给当下分组，无未来函数）
        :param raw_data: 原始数据
        :param valid_factors: 所需全部因子
        :param filter_mode: 过滤模式
        :param factors_setting: 因子配置
        :return: 处理后的数据
        """
        raw_data = self.add_industry(
            self.valid_data_filter(raw_data, valid_factors),
            self.industry_mapping,
            self.class_level
        )

        shifted_data = self.processor.shift_factors(
            raw_data, self.lag_period
        )

        filtered_data = self.filter(
            data=shifted_data,
            filter_mode=filter_mode,
            cycle=self.cycle
        ).run()

        processed_data = self.processor.preprocessing_factors_by_setting(
            filtered_data,
            factors_setting,
        )

        return processed_data

    # --------------------------
    # 计算指标
    # --------------------------
    def _calc_corr(
            self,
            processed_data: dict[str, pd.DataFrame],
            recent_period: int
    ) -> dict[str, pd.DataFrame]:
        """
        计算因子间相关系数
        :param processed_data: 预处理数据
        :param recent_period: 监控最近 n 期
        :return: 监控指标
        """
        return {
                "corr": self.evaluate.monitor.calc_corr(
                    processed_data, self.processed_factors_name
                ),
                "recent_corr": self.evaluate.monitor.calc_corr(
                    processed_data, self.processed_factors_name, recent_period
                ),
                "pairwise_corr": pd.DataFrame.from_dict(
                    {
                        factors_name:
                            self.evaluate.monitor.calc_pairwise_correlation(processed_data, factors_name, recent_period)
                        for factors_name in self.processed_factors_name
                    }
                )
            }

    def _calc_valuation_spread(
            self,
            processed_data: dict[str, pd.DataFrame],
            recent_period: int 
    ) -> dict[str, pd.DataFrame]:
        """
        计算监控指标
        :param processed_data: 预处理数据
        :param recent_period: 监控最近 n 期
        :return: 监控指标
        """
        metrics = {}
        ratio_metrics = defaultdict(list)

        for factor in self.processed_factors_name:
            for value in self.VALUE_SPREAD_PARAMETERS:
                metric_df = self.evaluate.monitor.calc_valuation_spread(
                    processed_data,
                    factor,
                    value,
                    recent_period
                )

                key = f"{factor}_valuation_spread_{value}"
                metrics[key] = metric_df

                ratio_metrics[f"valuation_spread_{value}"].append(
                    metric_df["ratio"].rename(factor)
                )

        ratio_metrics = {
            metric: pd.concat(ratios, axis=1)
            for metric, ratios in ratio_metrics.items()
        }

        return {
            **metrics,
            **ratio_metrics
        }

    def _calc_monitor_metrics(
            self,
            processed_data: dict[str, pd.DataFrame],
            recent_period: int
    ) -> dict[str, pd.DataFrame]:
        """
        计算监控指标
        :param processed_data: 预处理数据
        :param recent_period: 监控最近 n 期
        :return: 监控指标
        """
        # 预先生成函数映射关系
        metrics_map = {
            "turn": self.evaluate.monitor.calc_turnover_ratio,
            "cum": self.evaluate.monitor.calc_cum_return,
            "vol": self.evaluate.monitor.calc_volatility,
        }

        metrics = {}
        ratio_metrics = defaultdict(list)
        for metric, func in metrics_map.items():

            for factor in self.processed_factors_name:
                metric_df = func(processed_data, factor, recent_period)

                key = f"{factor}_{metric}"
                metrics[key] = metric_df
                ratio_metrics[metric].append(
                    metric_df["ratio"].rename(factor)
                )

        ratio_metrics = {
            metric: pd.concat(ratios, axis=1)
            for metric, ratios in ratio_metrics.items()
        }

        return {
            **metrics,
            **ratio_metrics
        }

    # --------------------------
    # 流程分析方法
    # --------------------------
    @validate_literal_params
    def _analyze(
            self,
            raw_data: dict[str, pd.DataFrame],
            filter_mode: FILTER_MODE,
    ) -> None:
        """
        单个因子分析
        :param raw_data: 原始数据
        :param filter_mode: 过滤模式
        """
        self.logger.info(f"start: {filter_mode}")
        # --------------------------
        # 初始化
        # --------------------------
        storage_dir = self._get_storage_dir(filter_mode)

        # --------------------------
        # 数据处理
        # --------------------------
        processed_data = self._data_process(
            raw_data, self.valid_factors_name, filter_mode, self.factors_setting
        )
        if not processed_data:
            raise ValueError("过滤数据为空值")

        # ---------------------------------------
        # 指标
        # ---------------------------------------
        result = {
            **self._calc_corr(processed_data, self.recent_period),
            **self._calc_monitor_metrics(processed_data, self.recent_period),
            **self._calc_valuation_spread(processed_data, self.recent_period),
        }

        # ---------------------------------------
        # 可视化
        # ---------------------------------------
        self._draw_charts(storage_dir, result, self.visual_setting)

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(self) -> None:
        """执行完整分析流程"""
        for filter_mode in get_args(self.filter_mode):
            self._analyze(
                raw_data=self.raw_data,
                filter_mode=filter_mode
            )
