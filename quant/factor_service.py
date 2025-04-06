"""
因子分析业务层
"""

from typing import Literal, get_args
from multiprocessing import Pool, Manager, Lock
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from pathlib import Path

import warnings
import traceback
import numpy as np
import pandas as pd

from base_service import BaseService
from range_filter import RangeFilter
from constant.quant import (
                            FactorVisualization, RESTRUCTURE_FACTOR,
                            NEGATIVE_SINGLE_COLUMN
                        )
from constant.path_config import DataPATH
from constant.type_ import (
    CYCLE, CLASS_LEVEL, FILTER_MODE,
    GROUP_MODE, validate_literal_params
)
from data_storage import DataStorage


warnings.filterwarnings("ignore")


#####################################################
class FactorAnalyzer(BaseService):
    """
    单因子分析：
        -1 IC（Rank）
        -2 收益率（分层、多空、T值）
        -3 比率
        -4 综合评估指标
    """

    DESCRIPTIVE_FACTOR = [
        "市值", "市净率"
    ]
    # "real_pctChg"
    CORE_FACTOR = [
        "对数市值", "close", "pctChg"
    ]

    @validate_literal_params
    def __init__(
            self,
            source_dir: str | Path,
            factors_name: list[str],
            cycle: CYCLE,
            class_level: CLASS_LEVEL = "一级行业",
            standardization: bool = True,
            mv_neutral: bool = False,
            industry_neutral: bool = False,
            restructure: bool = False,
            lag_period: int = 1,
            benchmark_code: str = "000300",
            double_sort_factor_name: str = '',
            group_nums: int = 5,
            processes_nums: int = 1
    ):
        """
        :param source_dir: 原始数据目录名称
        :param factors_name: 因子名
        :param cycle: 周期
        :param class_level: 行业级数
        :param standardization: 标准化
        :param mv_neutral: 市值中性化
        :param industry_neutral: 行业中性化
        :param restructure: 因子重构
        :param lag_period: 滞后期数
        :param benchmark_code: 基准指数代码
        :param double_sort_factor_name: 双重排序因子名
        :param group_nums: 分组数
        :param processes_nums: 多进程数
        """
        # 分析参数
        self.source_dir = source_dir
        self.factors_name = factors_name
        self.cycle = cycle
        self.class_level = class_level
        self.standardization = standardization
        self.mv_neutral = mv_neutral
        self.industry_neutral = industry_neutral
        self.restructure = restructure
        self.lag_period = lag_period
        self.benchmark_code = benchmark_code
        self.double_sort_factor_name = double_sort_factor_name
        self.group_nums = group_nums
        self.processes_nums = processes_nums

        # --------------------------
        # 初始化配置参数
        # --------------------------
        # 工具类
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
        self.storage_dir = DataPATH.QUANT_FACTOR_ANALYSIS_RESULT
        self.source_dir = DataPATH.QUANT_CONVERT_RESULT / self.source_dir

        # --------------------------
        # 数据
        # --------------------------
        self.industry_mapping = self.load_industry_mapping()
        self.listed_nums = self.load_listed_nums()
        self.raw_data = self.load_factor_data(self.source_dir)

        # --------------------------
        # 其他参数
        # --------------------------
        self.support_factors_name = self._get_support_factors()
        self.group_label = self.setup_group_label(self.group_nums)
        self.visual_setting = FactorVisualization()
        self.filter_mode = Literal[
            "_entire_filter",
            "_overall_filter",
            "_large_cap_filter",
            "_mega_cap_filter",
            "_small_cap_filter"
        ]
        self.group_mode = Literal["frequency"]

        # 最少行数
        self.min_nums = 100
    
    def _get_support_factors(
            self
    ) -> list[str]:
        """获取分析因子之外所需的因子"""
        factors = (
            [self.double_sort_factor_name]
            + self.DESCRIPTIVE_FACTOR
            + self.CORE_FACTOR
            + self.filter.PARAMETERS
        )
        return [f for f in set(factors) if f]
    
    # --------------------------
    # 单因子初始化
    # --------------------------
    def _get_storage_dir(
            self,
            factor_name: str,
            filter_mode: str 
    ) -> Path:
        """获取存储地址"""
        return (
                self.storage_dir
                / factor_name
                / f"股票池{filter_mode}-滞后{self.lag_period}{self.cycle[0]}-"
                  f"mve{self.mv_neutral}-ine{self.industry_neutral}-res{self.restructure}"
        )

    def _get_valid_factor(
            self,
            factor_name: str 
    ) -> list[str]:
        """获取有效因子"""
        # 添加重构因子
        factors_name = (
            [f for f in [factor_name, RESTRUCTURE_FACTOR.get(factor_name, "")] if f]
            if self.restructure
            else [factor_name]
        )
        # 添加必要因子
        return list(set(
            factors_name + self.support_factors_name
        ))

    # --------------------------
    # 数据处理
    # --------------------------
    @validate_literal_params
    def _data_process(
            self,
            raw_data: dict[str, pd.DataFrame],
            valid_factors: list[str],
            filter_mode: FILTER_MODE,
            group_mode: GROUP_MODE,
            factor_name: str,
            processed_factor_col: str, 
    ) -> dict[str, pd.DataFrame]:
        """
        数据处理 
            -1 截取（全部因子） -2 平移（除 pctChg 之外） -3 过滤（过滤因子） 
            -4 预处理（回测因子） -5 分组（预处理因子、使用过去的因子给当下分组，无未来函数）
        :param raw_data: 原始数据
        :param valid_factors: 所需全部因子
        :param filter_mode: 过滤模式
        :param group_mode: 分组模式
        :param factor_name: 因子名
        :param processed_factor_col: 预处理因子名
        :return: 处理后的数据
        """
        valid_data = self.valid_data_filter(raw_data, valid_factors)

        add_industry_data = self.add_industry(
            valid_data,
            self.industry_mapping,
            self.class_level
        )

        shifted_data = self.processor.shift_factors(
            add_industry_data, self.lag_period
        )

        filtered_data = self.filter(
            data=shifted_data,
            filter_mode=filter_mode,
            cycle=self.cycle
        ).run()

        filtered_data = {
            date: df
            for date, df in filtered_data.items()
            if df.shape[0] > self.min_nums
        }

        processed_data = self.processor.preprocessing_factor_data(
            filtered_data,
            factor_name,
            self.standardization,
            self.mv_neutral,
            self.industry_neutral,
            self.restructure
        )

        grouped_data = self.processor.divide_into_group(
            data=processed_data,
            factor_col=factor_name,
            processed_factor_col=processed_factor_col,
            group_mode=group_mode,
            group_nums=self.group_nums,
            group_label=self.group_label,
            negative=True if factor_name in NEGATIVE_SINGLE_COLUMN else False
        )
        return grouped_data

    # --------------------------
    # 指标计算
    # --------------------------
    def _get_measure_indicator(
            self,
            factor_name: str,
            filter_mode: str,
            group_mode: str,
            best_label: str,
            result: dict
    ) -> pd.DataFrame:
        """
        获取综合评价指标
        :param factor_name: 因子名
        :param filter_mode: 过滤模式
        :param group_mode: 分组模式
        :param best_label: 最优标签
        :param result: 回测结果
        :return: 综合评价指标
        """
        metrics = pd.Series(
            {
                "因子名": factor_name,
                "回测时长": result["basic_stats"].loc["value", "duration"],
                "研究范围": filter_mode,
                "周期": self.cycle,
                "标准化": self.standardization,
                "市值中性化": self.mv_neutral,
                "行业中性化": self.industry_neutral,
                "因子重构": self.restructure,
                "分组": group_mode,
                "ic_mean": result["ic_stats"]["ic_mean"].values[0],
                "ic_ir": result["ic_stats"]["ic_ir"].values[0],
                "ic_significance": result["ic_stats"]["ic_significance"].values[0],
                "秩相关系数": result["basic_stats"]["corr"].values[0],
                "JT_统计量": result["basic_stats"]["JT_统计量"].values[0],
                "JT_p值": result["basic_stats"]["JT_p值"].values[0],
                "最优组t值":
                    result["returns_stats"].loc["t_value", best_label],
                "多空组t值": result["returns_stats"].loc["t_value"].iloc[-1],
                "半衰期": result["ic_stats"]["half_life"].values[0]
            }
        )
        metrics["judgment"] = (
            False if (
                (abs(metrics["ic_mean"]) < 0.03)
                or (abs(metrics["ic_ir"]) < 0.5)
                or (metrics["ic_significance"] < 0.6)
                or (abs(metrics["秩相关系数"]) < 0.7)
                or (metrics["最优组t值"] < 2)
                or (metrics["多空组t值"] < 2)
            ) else True
        )
        
        return pd.DataFrame(metrics).T

    @classmethod
    def _judgment_factor(
            cls,
            metrics: pd.Series
    ) -> bool:
        """判定因子是否纳入因子池"""
        return False if (
                (abs(metrics["ic_mean"]) < 0.03)
                or (abs(metrics["ic_ir"]) < 0.5)
                or (metrics["ic_significance"] < 0.6)
                or (abs(metrics["秩相关系数"]) < 0.7)
                or (metrics["JT_p值"] > 0.05)
                or (metrics["最优组t值"] < 2)
                or (metrics["多空组t值"] < 2)
                or (metrics["半衰期"] < 6)
        ) else True

    # --------------------------
    # 可视化方法
    # --------------------------
    @classmethod
    def _calc_and_save_pdf(
            cls,
            data: dict[str, pd.DataFrame],
            factor_name: str,
            storage_dir: Path
    ) -> None:
        """
        概率密度函数计算及可视化
        :param data: 原始数据
        :param factor_name: 因子名
        :param storage_dir: 存储数据目录名称
        """
        try:
            # 获取最新数据集
            latest_key = list(data.keys())[-1]
            latest_df = data[latest_key]
            processed_col = f"processed_{factor_name}"

            # 创建图形对象
            fig, axes = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(40, 30),
                tight_layout=True
            )

            # 定义通用绘图函数
            def plot_kde(ax: plt.Axes, series: pd.Series, title: str) -> None:
                """绘制单个核密度估计图"""
                # 清理数据
                clean_data = series.dropna()
                if len(clean_data) < 2:
                    raise ValueError("有效数据点不足，无法计算分布")

                # 计算核密度估计
                kde = gaussian_kde(clean_data)
                eval_points = np.linspace(
                    clean_data.min(),
                    clean_data.max(),
                    num=500  # 增加点数使曲线更平滑
                )
                pdf_values = kde.pdf(eval_points)

                # 绘制图形
                ax.plot(eval_points, pdf_values, lw=3)
                ax.set_title(title, fontsize=35, pad=20)
                ax.tick_params(axis="both", labelsize=25)
                ax.grid(True, alpha=0.3)

            # 绘制原始分布
            try:
                original_series = latest_df[factor_name]
                plot_kde(axes[0], original_series, "Distribution: Original")
            except KeyError:
                raise KeyError(f"列 {factor_name} 不存在于数据中")

            # 绘制处理后的分布
            try:
                processed_series = latest_df[processed_col]
                plot_kde(axes[1], processed_series, "Distribution: Processed")
            except KeyError:
                raise KeyError(f"列 {processed_col} 不存在于数据中")

            # 保存并清理
            plt.savefig(storage_dir / "因子分布.png", bbox_inches="tight", dpi=100)
            plt.close(fig)

        except TypeError as e:
            print(f"数据类型错误: {str(e)}，无法计算分布")
        except ValueError as e:
            print(f"数据问题: {str(e)}")
        except Exception as e:
            print(f"未知错误: {str(e)}")

    def _save_measure_indicator(
            self,
            metrics: pd.DataFrame,
            lock: Lock
    ) -> None:
        """
        存储评估指标
        :param metrics: 评估指标
        :param lock: 进程锁
        """
        table_storage = DataStorage(self.storage_dir)
        if lock is None:
            table_storage.write_df_to_excel(
                metrics, "因子分类表", mode="a", index=False,
                subset=[
                    "因子名", "回测时长", "研究范围", "周期", "标准化",
                    "市值中性化", "行业中性化", "因子重构", "分组"
                ],
                keep="last"
            )
        else:
            with lock:
                table_storage.write_df_to_excel(
                    metrics, "因子分类表", mode="a", index=False,
                    subset=[
                        "因子名", "回测时长", "研究范围", "周期", "标准化",
                        "市值中性化", "行业中性化", "因子重构", "分组"
                    ],
                    keep="last"
                )

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

    def _store_results(
            self,
            model_data: dict[str, pd.DataFrame],
            storage_path: Path
    ) -> None:
        """存储分析结果"""
        model_data = self._reset_index(model_data)
        DataStorage(storage_path).write_dict_to_parquet(model_data)

    # --------------------------
    # 单因子分析流程
    # --------------------------
    @validate_literal_params
    def _analyze_single_factor(
            self,
            raw_data: dict[str, pd.DataFrame],
            factor_name: str,
            group_mode: GROUP_MODE,
            filter_mode: FILTER_MODE,
            lock: Lock = None
    ) -> None:
        """
        单个因子分析
        :param raw_data: 原始数据
        :param factor_name: 因子名
        :param group_mode: 分组模式
        :param filter_mode: 过滤模式
        """
        try:
            self.logger.info(f"start: {factor_name} - {group_mode} - {filter_mode}")
            # --------------------------
            # 初始化
            # --------------------------
            storage_dir = self._get_storage_dir(factor_name, filter_mode)
            processed_factor_col = f"processed_{factor_name}"
            valid_factors = self._get_valid_factor(factor_name)

            # --------------------------
            # 数据处理
            # --------------------------
            grouped_data = self._data_process(
                raw_data, valid_factors, filter_mode, group_mode, factor_name, processed_factor_col
            )
            if not grouped_data:
                raise ValueError("分组数据为空值")

            # ---------------------------------------
            # 指标 -1 覆盖度 -2 描述性参数 -3 因子指标 -4 收益率指标 -5 综合评价指标
            # ---------------------------------------
            ic_stats = self.calc_ic_metrics(
                    grouped_data, processed_factor_col, self.cycle
                )
            ic_mean = ic_stats["ic_stats"].loc["ic", "ic_mean"]
            reverse = True if ic_mean < 0 else False

            result = {
                **{
                    "coverage": self.calc_coverage(grouped_data, self.listed_nums),
                    "desc_stats": self.get_desc_stats(
                        grouped_data,
                        list(set([factor_name, processed_factor_col] + self.DESCRIPTIVE_FACTOR))
                    )
                },
                **ic_stats,
                **self.calc_return_metrics(
                    grouped_data, self.cycle, self.group_label,
                    reverse=reverse
                ),
                **self.calc_return_metrics(
                    grouped_data, self.cycle, self.group_label,
                    mode="mv_weight", reverse=reverse, prefix="mw"
                ),
            }

            measure_metrics = self._get_measure_indicator(
                factor_name, filter_mode, group_mode,
                self.group_label[0] if ic_mean < 0 else self.group_label[-1],
                result
            )

            # ---------------------------------------
            # 存储、可视化
            # ---------------------------------------
            # excel 因子判断
            self._save_measure_indicator(measure_metrics, lock)
            # pycharts IC 收益率
            self._draw_charts(storage_dir, result, self.visual_setting)
            # png 因子分布
            self._calc_and_save_pdf(grouped_data, factor_name, storage_dir)
            # parquet 分组数据
            # self._store_results(grouped_data, storage_dir)
        except Exception as e:
            self.logger.error(
                f"错误信息: {factor_name} {group_mode} {filter_mode}|"
                f"异常类型: {type(e).__name__}, 错误详情: {str(e)}, 堆栈跟踪:\n{traceback.format_exc()}"
            )

    # --------------------------
    # 多进程方法
    # --------------------------
    @classmethod
    @validate_literal_params
    def _parallel_task(
            cls,
            init_params: dict,
            factor_name: str,
            group_mode: GROUP_MODE,
            filter_mode: FILTER_MODE,
            lock: Lock,
            raw_data: dict[str, pd.DataFrame],
            industry_codes: pd.DataFrame,
            listed_nums: pd.Series
    ) -> None:
        """并行任务执行入口"""
        # 重建分析器实例
        analyzer = cls(
            source_dir=init_params['source_dir'],
            factors_name=init_params['factors_name'],
            cycle=init_params['cycle'],
            class_level=init_params['class_level'],
            standardization=init_params['standardization'],
            mv_neutral=init_params['mv_neutral'],
            industry_neutral=init_params['industry_neutral'],
            lag_period=init_params['lag_period'],
            benchmark_code=init_params['benchmark_code'],
            double_sort_factor_name=init_params['double_sort_factor_name'],
            group_nums=init_params['group_nums']
        )

        # 直接使用主进程加载的数据
        analyzer.industry_codes = industry_codes
        analyzer.listed_nums = listed_nums

        analyzer._analyze_single_factor(
            raw_data=raw_data,
            factor_name=factor_name,
            group_mode=group_mode,
            filter_mode=filter_mode,
            lock=lock
        )

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(self) -> None:
        """执行完整分析流程"""
        for k, factor_name in enumerate(self.factors_name, 1):
            for group_method in get_args(self.group_mode):
                for filter_mode in get_args(self.filter_mode):
                    self._analyze_single_factor(
                        raw_data=self.raw_data,
                        factor_name=factor_name,
                        group_mode=group_method,
                        filter_mode=filter_mode
                    )

    def multi_run(self) -> None:
        """多进程执行完整分析流程"""
        # 生成任务参数列表
        task_args = [
            (factor_name, group_mode, filter_mode)
            for factor_name in self.factors_name
            for group_mode in get_args(self.group_mode)
            for filter_mode in get_args(self.filter_mode)
        ]

        # 获取需要传递的初始化参数
        init_params = {
            'source_dir': str(self.source_dir),  # 转换为字符串保证可序列化
            'factors_name': self.factors_name,
            'cycle': self.cycle,
            'class_level': self.class_level,
            'standardization': self.standardization,
            'mv_neutral': self.mv_neutral,
            'industry_neutral': self.industry_neutral,
            'lag_period': self.lag_period,
            'benchmark_code': self.benchmark_code,
            'double_sort_factor_name': self.double_sort_factor_name,
            'group_nums': self.group_nums
        }

        # 创建进程池
        with Manager() as manager:
            # 创建共享锁和结果队列
            lock = manager.Lock()
            with Pool(processes=self.processes_nums) as pool:
                # 使用 starmap_async 支持进度跟踪
                results = pool.starmap_async(
                    self._parallel_task,
                    [(init_params, factor_name, group_mode, filter_mode, lock,
                      self.raw_data, self.industry_mapping, self.listed_nums)
                     for factor_name, group_mode, filter_mode in task_args]
                )
                results.get()
