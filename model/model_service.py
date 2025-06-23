"""模型 回测/预测 业务层"""
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

from constant.path_config import DataPATH
from constant.quant import ModelVisualization
from constant.type_ import CYCLE, validate_literal_params, FILTER_MODE
from utils.storage import DataStorage
from utils.quant_service import QuantService
from utils.stock_pool_filter import StockPoolFilter
from utils.processor import DataProcessor

import warnings
warnings.filterwarnings("ignore")


#####################################################
class ModelAnalyzer(QuantService):
    """模型分析"""

    CORE_FACTOR = [
        "对数市值", "open", "close", "unadjusted_close", "pctChg"
    ]
    DESCRIPTIVE_FACTOR = [
        "市值", "市净率"
    ]
    predict_date = datetime.strptime("2100-01-01", "%Y-%m-%d").date()
    
    @validate_literal_params
    def __init__(
            self,
            model: callable,
            model_setting: dataclass,
            filter_mode: FILTER_MODE,
            cycle: CYCLE,
            source_dir: str | Path,
            storage_dir: str | Path,
            benchmark_code: str = "000300"
    ):
        """
        :param model: 模型
        :param model_setting: 模型配置
        :param filter_mode: 标的池
        :param cycle: 周期
        :param source_dir: 原始数据目录名称
        :param storage_dir: 存储数据目录名称
        :param benchmark_code: 基准指数代码
        """
        self.storage_dir = storage_dir
        self.source_dir = source_dir
        self.model = model
        self.model_setting = model_setting
        self.cycle = cycle
        self.benchmark_code = benchmark_code
        self.filter_mode = filter_mode
        self.stock_pool_filter = StockPoolFilter                        # 标的池过滤类
        self.processor = DataProcessor()                                # 数据处理实例

        # --------------------------
        # 初始化配置参数
        # --------------------------
        # 路径
        self.storage_dir = DataPATH.QUANT_MODEL_ANALYSIS_RESULT / self.storage_dir
        self.source_dir = DataPATH.QUANT_CONVERT_RESULT / self.source_dir

        # 数据
        self.industry_mapping = self.load_industry_mapping()
        self.listed_nums = self.load_listed_nums()
        self.raw_data = self.load_factors_value_by_dask(self.source_dir)

        # 过滤因子
        self.model_setting.factors_setting = self._factor_filter()

        # 因子名
        self.model_factors_name = self._get_model_factors_name()
        self.backtest_factors_name = self._get_backtest_factors_name()

        # 其他参数
        self.visual_setting = ModelVisualization()
        self.model_setting.group_label = self.setup_group_label(self.model_setting.group_nums)

        # 日志
        self.logger = self.setup_logger(self.storage_dir)


    def _factor_filter(
            self
    ) -> list:
        """
        因子过滤
            -1 一级/二级分类因子
            -2 半衰期
            -3 因子适配池
        """
        if not self.model_setting.factor_filter:
            return self.model_setting.factors_setting

        result = []
        primary_cats = self.model_setting.factor_primary_classification or []
        secondary_cats = self.model_setting.factor_secondary_classification or []
        half_life_range = self.model_setting.factor_half_life or (0, float('inf'))
        filter_modes = self.model_setting.factor_filter_mode or set()

        for setting in self.model_setting.factors_setting:
            # 检查因子类型
            classification_ok = (
                    setting.primary_classification in primary_cats or
                    setting.secondary_classification in secondary_cats
            ) if primary_cats or secondary_cats else True

            # 检查半衰期是否在有效范围内
            half_life_ok = (half_life_range[0] <= setting.half_life <= half_life_range[1])

            # 检查过滤模式匹配性
            filter_mode_ok = setting.filter_mode in filter_modes if filter_modes else True

            # 组合条件（分类满足 AND 半衰期 AND 过滤模式）
            if classification_ok and half_life_ok and filter_mode_ok:
                result.append(setting)

        return result

    def _get_model_factors_name(
            self
    ) -> list[str]:
        """获取因子名"""
        return [
            setting.factor_name
            for setting in self.model_setting.factors_setting
        ]

    def _get_backtest_factors_name(
            self
    ) -> list[str]:
        """
        获取回测全部所需的因子
            -1 模型因子
            -2 支持因子
        """
        support_factors = [f for f in set(
                self.DESCRIPTIVE_FACTOR
                + self.CORE_FACTOR
                + self.stock_pool_filter.PARAMETERS
                + ["date", "股票代码"]
        )]

        backtest_factor = list(set(
            self.model_factors_name
            + support_factors
        ))
        # 排序（支持复现）
        backtest_factor.sort()

        return backtest_factor

    # --------------------------
    # 数据处理
    # --------------------------
    def _pre_processing(
            self,
            raw_data: pd.DataFrame,
            backtest_factors: list[str]
    ) -> pd.DataFrame:
        """
        模型前数据预处理
            -1 因子过滤
            -2 非因子数据添加 -> 所属行业/预测日期数据
            -3 平移（除 pctChg/date/股票代码/行业）
            -4 标的池过滤
            -5 有效数据量过滤 -> 单日数据量高于分组数

        :param raw_data: 原始数据
        :param backtest_factors: 回测所需全部因子
        :return: 预处理后的数据
        """
        return (
            raw_data
            .pipe(self.valid_factors_filter,
                  valid_factors=backtest_factors)
            .pipe(self.add_industry,
                  industry_mapping=self.industry_mapping,
                  class_level=self.model_setting.class_level)
            .pipe(self._add_the_latest_df)
            .pipe(self.processor.refactor.shift_factors_value,
                  fixed_col=["股票代码", "行业", "pctChg", "date"],
                  lag_periods=self.model_setting.lag_period)
            .pipe(self.stock_pool_filter(filter_mode=self.filter_mode, cycle=self.cycle))
            .pipe(self._rows_quantity_check)
            .pipe(self.time_continuity_test, cycle=self.cycle)
        )

    def _add_the_latest_df(
            self,
            input_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        创建最新日期的数据
        :param input_df: 平移前的数据
        """
        latest_date = input_df["date"].unique()[-1]

        df = input_df[input_df["date"] == latest_date].copy(deep=True)
        df["pctChg"] = np.nan
        df["date"] = self.predict_date

        return pd.concat([input_df, df])

    def _rows_quantity_check(
            self,
            input_df: pd.DataFrame
    ) -> pd.DataFrame:
        """行数检查 -> 行数大于分组数"""
        # 按日期分组处理
        grouped = input_df.groupby('date')
        result_dfs = []

        for date, group_df in grouped:
            # 检查当前日期组行数是否满足要求
            if len(group_df) >= self.model_setting.group_nums:
                result_dfs.append(group_df)
            else:
                print(f"{date} | 行数少于 {self.model_setting.group_nums}")

        # 合并结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    # --------------------------
    # 计算指标
    # --------------------------
    def _calc_model_metrics(
            self,
            grouped_data: dict[str, pd.DataFrame],
            ic_test: bool = False
    ) -> dict:
        """计算模型指标"""
        # ic检验条件：合成综合Z值
        if ic_test:
            ic_stats = self.calc_model_ic_metrics(
                grouped_data, "综合Z值", self.cycle
            )
            ic_mean = ic_stats["ic_stats"].loc["ic", "ic_mean"]
            reverse = True if ic_mean < 0 else False
        else:
            ic_stats = {}
            reverse = False

        return {
            "coverage": self.calc_coverage(grouped_data, self.listed_nums),
            **ic_stats,
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                reverse=reverse, prefix="0.0"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="mv_weight", reverse=reverse, prefix="0.0_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="position_weight", reverse=reverse, prefix="0.0_pw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                reverse=reverse, trade_cost=0.01, prefix="0.01"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="mv_weight", reverse=reverse, trade_cost=0.01, prefix="0.01_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="position_weight", reverse=reverse, trade_cost=0.01, prefix="0.01_pw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                reverse=reverse, trade_cost=0.03, prefix="0.03"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="mv_weight", reverse=reverse, trade_cost=0.03, prefix="0.03_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="position_weight", reverse=reverse, trade_cost=0.03, prefix="0.03_pw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                reverse=reverse, trade_cost=0.05, prefix="0.05"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="mv_weight", reverse=reverse, trade_cost=0.05, prefix="0.05_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="position_weight", reverse=reverse, trade_cost=0.05, prefix="0.05_pw"
            ),
        }

    # --------------------------
    # 存储、可视化 方法
    # --------------------------
    def _reset_index(
            self,
            model_data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """重置列名，代码 -> 企业简称"""
        # 企业简称映射字典
        short_name_mapping: dict = (
            self.industry_mapping
            .set_index("股票代码")
            ["公司简称"]
            .to_dict()
        )
        return {
            date: df_.rename(index=short_name_mapping)
            for date, df_ in model_data.items()
        }

    def _store_grouped_data(
            self,
            model_data: dict[str, pd.DataFrame],
    ) -> None:
        """存储模型分组数据"""
        model_data = self._reset_index(model_data)
        DataStorage(self.storage_dir).write_dict_to_parquet(
            model_data,
            merge_original_data=False
        )

    def _store_model_setting(
            self
    ) -> None:
        """存储每一期的筛选后的因子"""
        with open(
                f"{self.storage_dir}/model_setting.yaml",
                "w",
                encoding="utf-8"
        ) as f:
            yaml.safe_dump(asdict(self.model_setting), f, allow_unicode=True)

    def _store_selected_factors(
            self,
            selected_factors: dict[str, list[str]]
    ) -> None:
        """存储每一期的筛选后的因子"""
        with open(
                f"{self.storage_dir}/selected_factors.yaml",
                "w",
                encoding="utf-8"
        ) as f:
            yaml.safe_dump(selected_factors, f, allow_unicode=True)

    # --------------------------
    # 流程分析方法
    # --------------------------
    @validate_literal_params
    def _analyze(
            self
    ) -> None:
        """执行完整分析流程"""
        self.logger.info("---------- 模型前数据预处理 ----------")
        pre_processing_df = self._pre_processing(
            self.raw_data,
            self.backtest_factors_name
        )
        if pre_processing_df.empty:
            raise ValueError("过滤数据为空值")

        # ---------------------------------------
        # 生成模型
        # ---------------------------------------
        self.logger.info("---------- 模型生成 ----------")
        model = self.model(
            input_df=pre_processing_df,
            model_setting=self.model_setting
        )
        model_df, metrics_df = model.run()
        model_data = {
            str(date): group
            for date, group in model_df.groupby("date")
        }

        # ---------------------------------------
        # 模型评估
        # ---------------------------------------
        self.logger.info("---------- 模型评估 ----------")
        result = {
            **self._calc_model_metrics(
                model_data,
                ic_test=True if "综合Z值" in model_df.columns else False
            ),
            **{"模型评估指标": metrics_df}
        }

        # ---------------------------------------
        # 存储、可视化
        # ---------------------------------------
        self.logger.info("---------- 结果存储、可视化 ----------")
        self._draw_charts(self.storage_dir, result, self.visual_setting)
        self._store_grouped_data(model_data)
        # self._store_selected_factors(selected_factors)

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(self) -> None:
        """执行完整分析流程"""
        self.logger.info(f"start: {str(self.model)} | {self.filter_mode}")

        # 存储模型信息
        self._store_model_setting()

        # 执行分析
        self._analyze()
