"""
因子分析业务层
"""

from typing import Literal, get_args
from multiprocessing import Pool, Manager, Lock
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from pathlib import Path

import warnings
import numpy as np
import pandas as pd

from quant_service import QuantService
from stock_pool_filter import StockPoolFilter
from check_factor_feature import DifferentMarketAnalyzer
from constant.quant import (
                            Factor, NEGATIVE_SINGLE_COLUMN
                        )
from constant.path_config import DataPATH
from constant.type_ import (
    CYCLE, CLASS_LEVEL, FILTER_MODE,
    GROUP_MODE, validate_literal_params
)
from utils.storage import DataStorage
from utils.processor import DataProcessor


warnings.filterwarnings("ignore")


#####################################################
class FactorAnalyzer(QuantService):
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
            index_code: str,
            factors_name: list[str],
            cycle: CYCLE,
            class_level: CLASS_LEVEL = "一级行业",
            standardization: bool = True,
            transfer_mode: str | None = None,
            mv_neutral: bool = False,
            industry_neutral: bool = False,
            lag_period: int = 1,
            benchmark_code: str = "000300",
            double_sort_factor_name: str = '',
            group_nums: int = 5,
            group_mode: GROUP_MODE = "frequency",
            processes_nums: int = 1
    ):
        """
        :param source_dir: 原始数据目录名称
        :param index_code: 指数代码
        :param factors_name: 因子名
        :param cycle: 周期
        :param class_level: 行业级数
        :param transfer_mode: 变换模式 -1 box_cox -2 yeo_johnson
        :param standardization: 标准化
        :param mv_neutral: 市值中性化
        :param industry_neutral: 行业中性化
        :param lag_period: 滞后期数
        :param benchmark_code: 基准指数代码
        :param double_sort_factor_name: 双重排序因子名
        :param group_mode: 分组模式
        :param group_nums: 分组数
        :param processes_nums: 多进程数
        """
        # 分析参数
        self.source_dir = source_dir
        self.index_code = index_code
        self.factors_name = factors_name
        self.cycle = cycle
        self.class_level = class_level
        self.transfer_mode = transfer_mode
        self.standardization = standardization
        self.mv_neutral = mv_neutral
        self.industry_neutral = industry_neutral
        self.lag_period = lag_period
        self.benchmark_code = benchmark_code
        self.double_sort_factor_name = double_sort_factor_name
        self.group_mode = group_mode
        self.group_nums = group_nums
        self.processes_nums = processes_nums

        # --------------------------
        # 初始化配置参数
        # --------------------------
        # 工具类
        self.processor = DataProcessor()                        # 数据处理实例
        self.stock_pool_filter = StockPoolFilter                # 标的池过滤类
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
        self.raw_data = self.load_factors_value_by_dask(self.source_dir)

        # --------------------------
        # 其他参数
        # --------------------------
        self.support_factors_name = self._get_support_factors()
        self.group_label = self.setup_group_label(self.group_nums)
        self.setting = Factor()
        self.filter_mode = Literal[
            # "_white_filter",
            # "_entire_filter",
            # "_overall_filter",
            # "_large_cap_filter",
            # "_mega_cap_filter",
            "_small_cap_filter"
        ]

        # --------------------------
        # 指数数据
        # --------------------------
        self.index_month_data = self._calculate_kline(
            self.load_index_kline(DataPATH.INDEX_KLINE_DATA / "month" / self.index_code),
            "month",
            self.setting.kline.kline
        ).dropna(how="any")
        self.index_day_data = self._calculate_kline(
            self.load_index_kline(DataPATH.INDEX_KLINE_DATA / "day" / self.index_code),
            "day",
            self.setting.kline.kline
        )

    def _get_support_factors(
            self
    ) -> list[str]:
        """获取分析因子之外所需的因子"""
        factors = (
            [self.double_sort_factor_name]
            + self.DESCRIPTIVE_FACTOR
            + self.CORE_FACTOR
            + self.stock_pool_filter.PARAMETERS
            + ["date", "股票代码"]
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
                / f"股票池{filter_mode}-{self.cycle}-"
                  f"trans_{self.transfer_mode}"
                  f"-mve_{self.mv_neutral}"
                  f"-ind_{self.industry_neutral}"
        )

    def _get_valid_factor(
            self,
            factor_name: str 
    ) -> list[str]:
        """获取有效因子"""
        # 添加必要因子
        return list(set(
            [factor_name] + self.support_factors_name
        ))

    # --------------------------
    # 数据处理
    # --------------------------
    @validate_literal_params
    def _pre_processing(
            self,
            raw_data: pd.DataFrame,
            backtest_factors: list[str],
            filter_mode: FILTER_MODE,
            group_mode: GROUP_MODE,
            factor_name: str,
            processed_factor_col: str, 
    ) -> pd.DataFrame:
        """
        数据处理 
            -1 截取（全部因子）
            -2 断点检查（平移之前若有断点，则影响因子平移，若之后再出现断点，则不影响回测整体效果，只是部分时点缺失）
            -3 平移（除 pctChg 之外）
            -4 过滤（过滤因子）
            -5 预处理（回测因子）
            -6 分组（预处理因子、使用过去的因子给当下分组，无未来函数）
        :param raw_data: 原始数据
        :param backtest_factors: 回测所需全部因子
        :param filter_mode: 过滤模式
        :param group_mode: 分组模式
        :param factor_name: 因子名
        :param processed_factor_col: 预处理因子名
        :return: 处理后的数据
        """
        return (
            raw_data
            .pipe(self.valid_factors_filter,
                  valid_factors=backtest_factors)
            .pipe(self.time_continuity_test,
                  cycle=self.cycle)
            .pipe(self.add_industry,
                  industry_mapping=self.industry_mapping,
                  class_level=self.class_level)
            .pipe(self.processor.refactor.shift_factors_value,
                  fixed_col=["股票代码", "行业", "pctChg", "date"],
                  lag_periods=self.lag_period)
            .pipe(self.stock_pool_filter(filter_mode=filter_mode, cycle=self.cycle))
            .pipe(self._quantity_check)
            .pipe(self._preprocessing_factor_data,
                  factor_name=factor_name)
            .pipe(self.processor.classification.divide_into_group,
                  factor_col=factor_name,
                  processed_factor_col=processed_factor_col,
                  group_mode=group_mode,
                  group_nums=self.group_nums,
                  group_label=self.group_label,
                  negative=True if factor_name in NEGATIVE_SINGLE_COLUMN else False)
        )

    def _quantity_check(
            self,
            input_df: pd.DataFrame
    ) -> pd.DataFrame:
        """行数检查 -> 行数大于分组数"""
        # 按日期分组处理
        grouped = input_df.groupby('date')
        result_dfs = []

        for date, group_df in grouped:
            # 检查当前日期组行数是否满足要求
            if len(group_df) >= self.group_nums:
                result_dfs.append(group_df)
            else:
                print(f"{date} | 行数少于 {self.group_nums}")

        # 合并结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    def _preprocessing_factor_data(
            self,
            input_df: pd.DataFrame,
            factor_name: str,
            prefix: str = "processed"
    ) -> pd.DataFrame:
        """
        数据预处理
            -1 缩尾
            -2 标准化
            -3 中性化
            -4 缩尾
            -5 标准化
        :param input_df: 初始数据
        :param factor_name: 因子名
        :param prefix: 预处理生成因子前缀
        :return: 处理过的数据
        """
        def __process_single_date(
                df_: pd.DataFrame,
        ) -> pd.DataFrame:
            """单日数据处理"""
            processed_col = f"{prefix}_{factor_name}"
            df_[processed_col] = df_[factor_name].copy()

            # -1 正态化变换
            if self.transfer_mode:
                df_[processed_col] = (
                    self.processor.refactor.box_cox_transfer(df_[processed_col]) if self.transfer_mode == "box_cox" else
                    self.processor.refactor.yeo_johnson_transfer(df_[processed_col])
                )

            # -2 第一次 去极值、标准化
            df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
            if self.standardization:
                df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

            # -3 中性化
            if self.mv_neutral:
                df_[processed_col] = self.processor.neutralization.log_market_cap(
                    df_[processed_col],
                    df_["对数市值"],
                    winsorizer=self.processor.winsorizer.percentile,
                    dimensionless=self.processor.dimensionless.standardization
                )
            if self.industry_neutral:
                df_[processed_col] = self.processor.neutralization.industry(
                    df_[processed_col],
                    df_["行业"]
                )

            # -4 第二次 去极值、标准化
            df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
            if self.standardization:
                df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

            return df_

        result_dfs = []
        for date, group_df in input_df.groupby("date"):
            try:
                processed_df = __process_single_date(group_df)
                result_dfs.append(processed_df)
            except Exception as e:
                self.logger.info(f"数据预处理流程有误: {date} | {e}")
                continue

        # 合并处理结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

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
                "正态变换": self.transfer_mode,
                "市值中性化": self.mv_neutral,
                "行业中性化": self.industry_neutral,
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
                "半衰期": result["ic_stats"]["half_life"].values[0],
                "牛市IC": result["different_market_result"].loc["Bull", "mean"],
                "牛市IR": result["different_market_result"].loc["Bull", "ic_ir"],
                "牛市t值": result["different_market_result"].loc["Bull", "t_stat"],
                "熊市IC": result["different_market_result"].loc["Bear", "mean"],
                "熊市IR": result["different_market_result"].loc["Bear", "ic_ir"],
                "熊市t值": result["different_market_result"].loc["Bear", "t_stat"],
                "震荡市IC": result["different_market_result"].loc["Range", "mean"],
                "震荡市IR": result["different_market_result"].loc["Range", "ic_ir"],
                "震荡市t值": result["different_market_result"].loc["Range", "t_stat"],
                "倒J型因子": result["basic_stats"].loc["value", "j_shape_p_value"],
            }
        )

        # 因子适用市场
        metrics["适用市场"] = ""
        if abs(metrics["牛市IC"]) >= 0.03 and abs(metrics["牛市IR"]) >= 0.5 and abs(metrics["牛市t值"]) >= 2:
            metrics["适用市场"] += "牛市/"
        if abs(metrics["熊市IC"]) >= 0.03 and abs(metrics["熊市IR"]) >= 0.5 and abs(metrics["熊市t值"]) >= 2:
            metrics["适用市场"] += "熊市/"
        if abs(metrics["震荡市IC"]) >= 0.03 and abs(metrics["震荡市IR"]) >= 0.5 and abs(metrics["震荡市t值"]) >= 2:
            metrics["适用市场"] += "震荡市"

        # 因子判定: -1 阿尔法因子 -2 动态因子
        if (
                abs(metrics["ic_mean"]) >= 0.03
                and abs(metrics["ic_ir"]) >= 0.5
                and metrics["ic_significance"] >= 0.6
                and abs(metrics["秩相关系数"]) >= 0.7
                and metrics["最优组t值"] >= 2
                and metrics["多空组t值"] >= 2
                and metrics["JT_p值"] <= 0.05
        ):
            metrics["judgment"] = "alpha"
        elif (
                abs(metrics["秩相关系数"]) >= 0.7
                and metrics["最优组t值"] >= 2
                and metrics["多空组t值"] >= 2
                and metrics["JT_p值"] <= 0.05
        ):
            metrics["judgment"] = "nonlinear"
        elif (
                abs(metrics["ic_ir"]) >= 0.5
                and metrics["ic_significance"] >= 0.6
                and abs(metrics["秩相关系数"]) >= 0.7
                and metrics["最优组t值"] >= 2
                and metrics["多空组t值"] >= 2
        ):
            metrics["judgment"] = "weak"
        else:
            metrics["judgment"] = "invalid"

        return pd.DataFrame(metrics).T

    # --------------------------
    # 可视化方法
    # --------------------------
    @classmethod
    def _calc_and_save_pdf(
            cls,
            data: dict[str, pd.DataFrame],
            factor_name: str,
            transfer_factor_name: str,
            storage_dir: Path,
            png_name: str = "因子分布"
    ) -> None:
        """
        概率密度函数计算及可视化
        :param data: 原始数据
        :param factor_name: 因子名
        :param transfer_factor_name: 变换因子名
        :param storage_dir: 存储数据目录名称
        :param png_name: 图片文件名
        """
        try:
            # 获取最新数据集
            latest_key = list(data.keys())[-1]
            latest_df = data[latest_key]

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
                processed_series = latest_df[transfer_factor_name]
                plot_kde(axes[1], processed_series, "Distribution: Processed")
            except KeyError:
                raise KeyError(f"列 {transfer_factor_name} 不存在于数据中")

            # 保存并清理
            plt.savefig(storage_dir / f"{png_name}.png", bbox_inches="tight", dpi=100)
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
                metrics,
                "因子分类表",
                sheet_name=metrics["judgment"][0],
                mode="a",
                index=False,
                subset=[
                    "因子名", "回测时长", "研究范围", "周期", "标准化", "正态变换",
                    "市值中性化", "行业中性化", "分组"
                ],
                keep="last"
            )
        else:
            with lock:
                table_storage.write_df_to_excel(
                    metrics,
                    "因子分类表",
                    sheet_name=metrics["judgment"][0],
                    mode="a",
                    index=False,
                    subset=[
                        "因子名", "回测时长", "研究范围", "周期", "标准化", "正态变换",
                        "市值中性化", "行业中性化", "分组"
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
            raw_data: pd.DataFrame,
            factor_name: str,
            filter_mode: FILTER_MODE,
            lock: Lock = None
    ) -> None:
        """
        单个因子分析
        :param raw_data: 原始数据
        :param factor_name: 因子名
        :param filter_mode: 过滤模式
        """
        self.logger.info(f"start: {factor_name} - {self.group_mode} - {filter_mode}")
        # --------------------------
        # 初始化
        # --------------------------
        storage_dir = self._get_storage_dir(factor_name, filter_mode)
        processed_factor_col = f"processed_{factor_name}"
        valid_factors = self._get_valid_factor(factor_name)

        # --------------------------
        # 数据处理
        # --------------------------
        preprocessing_df = self._pre_processing(
            raw_data,
            valid_factors,
            filter_mode,
            self.group_mode,
            factor_name,
            processed_factor_col
        )
        preprocessing_dict = {
            str(date): group.drop("date", axis=1)
            for date, group in preprocessing_df.groupby("date")
        }
        self.logger.info(f"数据处理完毕")

        # ---------------------------------------
        # 指标 -1 覆盖度 -2 描述性参数 -3 因子指标 -4 收益率指标 -5 马尔科夫链划分市场 -6 综合评价指标
        # ---------------------------------------
        # ic类统计
        ic_stats = self.calc_ic_metrics(
            preprocessing_dict,
            processed_factor_col,
            self.cycle
        )
        # ic均值
        ic_mean = ic_stats["ic_stats"].loc["ic", "ic_mean"]
        # 是否为反转因子
        reverse = True if ic_mean < 0 else False

        result = {
            **{
                "coverage": self.calc_coverage(preprocessing_dict, self.listed_nums),
                "desc_stats": self.get_desc_stats(
                    preprocessing_dict,
                    list(set([factor_name, processed_factor_col] + self.DESCRIPTIVE_FACTOR))
                ),
            },
            **ic_stats,
            **self.calc_return_metrics(
                preprocessing_dict,
                self.cycle,
                self.group_label,
                reverse=reverse
            ),
            **self.calc_return_metrics(
                preprocessing_dict,
                self.cycle,
                self.group_label,
                mode="mv_weight", reverse=reverse, prefix="mw"
            ),
        }

        # 识别不同市场下的因子表现
        dm_result = DifferentMarketAnalyzer(
            factor_ic=result["ic"]["ic"],
            month_market_metrics=self.index_month_data,
            day_market_metrics=self.index_day_data,
            cycle=self.cycle
        ).run()
        result.update(
            {"different_market_result": dm_result}
        )

        # 因子综合评价
        measure_metrics = self._get_measure_indicator(
            factor_name,
            filter_mode,
            self.group_mode,
            self.group_label[0] if ic_mean < 0 else self.group_label[-1],
            result
        )

        # ---------------------------------------
        # 存储、可视化
        # ---------------------------------------
        # excel 因子判断
        self._save_measure_indicator(
            measure_metrics,
            lock
        )
        # pycharts IC 收益率
        self._draw_charts(
            storage_dir,
            result,
            self.setting.visualization
        )
        # png 因子分布
        self._calc_and_save_pdf(
            preprocessing_dict,
            factor_name=factor_name,
            transfer_factor_name=f"processed_{factor_name}",
            storage_dir=storage_dir
        )
        self.logger.info(f"单因子完成回测")
        # parquet 分组数据
        # self._store_results(grouped_data, storage_dir)

    @validate_literal_params
    def _analyze_single_factor_debug(
            self,
            raw_data: pd.DataFrame,
            factor_name: str,
            filter_mode: FILTER_MODE,
            lock: Lock = None
    ) -> None:
        """
        单个因子分析
        :param raw_data: 原始数据
        :param factor_name: 因子名
        :param filter_mode: 过滤模式
        """
        self.logger.info(f"start: {factor_name} - {self.group_mode} - {filter_mode}")
        # --------------------------
        # 初始化
        # --------------------------
        storage_dir = self._get_storage_dir(factor_name, filter_mode)
        processed_factor_col = f"processed_{factor_name}"
        valid_factors = self._get_valid_factor(factor_name)

        # --------------------------
        # 数据处理
        # --------------------------
        preprocessing_df = self._pre_processing(
            raw_data,
            valid_factors,
            filter_mode,
            self.group_mode,
            factor_name,
            processed_factor_col
        )
        preprocessing_dict = {
            str(date): group
            for date, group in preprocessing_df.groupby("date")
        }

        # ---------------------------------------
        # 指标 -1 覆盖度 -2 描述性参数 -3 因子指标 -4 收益率指标 -5 马尔科夫链划分市场 -6 综合评价指标
        # ---------------------------------------
        # ic类统计
        ic_stats = self.calc_ic_metrics(
            preprocessing_dict,
            processed_factor_col,
            self.cycle
        )
        # ic均值
        ic_mean = ic_stats["ic_stats"].loc["ic", "ic_mean"]
        # 是否为反转因子
        reverse = True if ic_mean < 0 else False

        result = {
            **{
                "coverage": self.calc_coverage(preprocessing_dict, self.listed_nums),
                "desc_stats": self.get_desc_stats(
                    preprocessing_dict,
                    list(set([factor_name, processed_factor_col] + self.DESCRIPTIVE_FACTOR))
                ),
            },
            **ic_stats,
            **self.calc_return_metrics(
                preprocessing_dict,
                self.cycle,
                self.group_label,
                reverse=reverse
            ),
            **self.calc_return_metrics(
                preprocessing_dict,
                self.cycle,
                self.group_label,
                mode="mv_weight", reverse=reverse, prefix="mw"
            ),
        }

        # 识别不同市场下的因子表现
        dm_result = DifferentMarketAnalyzer(
            factor_ic=result["ic"]["ic"],
            month_market_metrics=self.index_month_data,
            day_market_metrics=self.index_day_data,
            cycle=self.cycle
        ).run()
        result.update(
            {"different_market_result": dm_result}
        )

        # 因子综合评价
        measure_metrics = self._get_measure_indicator(
            factor_name,
            filter_mode,
            self.group_mode,
            self.group_label[0] if ic_mean < 0 else self.group_label[-1],
            result
        )

        # ---------------------------------------
        # 存储、可视化
        # ---------------------------------------
        # excel 因子判断
        self._save_measure_indicator(
            measure_metrics,
            lock
        )
        # pycharts IC 收益率
        self._draw_charts(
            storage_dir,
            result,
            self.setting.visualization
        )
        # png 因子分布
        self._calc_and_save_pdf(
            preprocessing_dict,
            factor_name=factor_name,
            transfer_factor_name=f"processed_{factor_name}",
            storage_dir=storage_dir
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
            for filter_mode in get_args(self.filter_mode):
                self._analyze_single_factor(
                    raw_data=self.raw_data,
                    factor_name=factor_name,
                    filter_mode=filter_mode
                )

    def debug(self) -> None:
        """执行完整分析流程"""
        for k, factor_name in enumerate(self.factors_name, 1):
            for filter_mode in get_args(self.filter_mode):
                self._analyze_single_factor_debug(
                    raw_data=self.raw_data,
                    factor_name=factor_name,
                    filter_mode=filter_mode
                )

    def multi_run(self) -> None:
        """多进程执行完整分析流程"""
        # 生成任务参数列表
        task_args = [
            (factor_name, filter_mode)
            for factor_name in self.factors_name
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
                    [(init_params, factor_name, filter_mode, lock,
                      self.raw_data, self.industry_mapping, self.listed_nums)
                     for factor_name, filter_mode in task_args]
                )
                results.get()
