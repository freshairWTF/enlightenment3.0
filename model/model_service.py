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
        "对数市值", "open", "close", "pctChg"
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
        half_life_range = self.model_setting.factor_half_life or (0, float("inf"))
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
        grouped = input_df.groupby("date")
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
    @classmethod
    def _calc_descriptive_factors(
            cls,
            model_df: pd.DataFrame,
            cycle: str = ""
    ) -> pd.DataFrame:
        """
        计算分组描述性统计均值
        :param model_df: 模型运行结果（需包含 date, group, 市值, 市净率, position_weight 列）
        :param cycle: 统计周期
            "M" : 按月统计
            "Y" : 按年统计
            "ALL": 全部统计
        :return: 分组描述性统计DataFrame
        """
        # -1 预处理
        df = model_df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # -2 创建临时加权计算列
        df['市值_weighted'] = df['市值'] * df['position_weight']
        df['市净率_weighted'] = df['市净率'] * df['position_weight']

        # -3 根据周期选择分组方式
        if cycle == 'ALL':
            grouped = df.groupby("group")
        elif cycle:
            grouped = df.groupby([
                "group",
                pd.Grouper(key='date', freq=cycle)
            ])
        else:
            grouped = df.groupby(["group", "date"])

        # -4 聚合计算
        descriptive = grouped.agg(
            市值=("市值", "mean"),
            市值_weighted_sum=("市值_weighted", "sum"),
            市净率=("市净率", "mean"),
            市净率_weighted_sum=("市净率_weighted", "sum"),
            weight_sum=("position_weight", "sum")           # 权重总和用于加权平均
        ).reset_index()
        descriptive["仓位权重加权市值"] = descriptive["市值_weighted_sum"] / descriptive["weight_sum"]
        descriptive["仓位权重加权市净率"] = descriptive["市净率_weighted_sum"] / descriptive["weight_sum"]

        # -5 单位转换与清理
        descriptive["市值"] /= 10 ** 8
        descriptive["仓位权重加权市值"] /= 10 ** 8
        descriptive.drop(
            columns=['市值_weighted_sum', '市净率_weighted_sum', 'weight_sum'],
            inplace=True
        )

        return descriptive.dropna(
            subset=["市值", "市净率", "仓位权重加权市值", "仓位权重加权市净率"],
            how="all"
        )

    @classmethod
    def _calc_trading_stats(
            cls,
            model_df: pd.DataFrame,
            commission_rate: float = 0.0003,
            stamp_tax_rate: float = 0.001,
            transfer_rate: float = 0.00001,
            slippage_rate: float = 0.003,
            cycle: str = ""
    ) -> pd.DataFrame:
        """
        计算模型（分组）交易数据
        :param model_df: 模型结果
        :param commission_rate: 佣金费率（默认万3）
        :param stamp_tax_rate: 印花税率（默认千1）
        :param transfer_rate: 过户费率（默认万0.1，沪市）
        :param slippage_rate: 滑点（默认千3）
        :param cycle: 统计周期
            "M" : 按月统计
            "Y" : 按年统计
            "ALL": 全部统计
        """
        # -1 计算交易统计数据
        all_stats = []
        for group_name, group in model_df.groupby("group"):
            df = group.pivot(
                index="date",
                columns="股票代码",
                values="position_weight"
            ).fillna(0)
            # -1 每期持股数量
            hold_counts = df.mask(df != 0).sum(axis=1)
            # -2 最大仓位
            max_weights = df.max(axis=1)
            # -3 最小仓位（排除0值）
            min_weights = df.mask(df <= 0).min(axis=1).fillna(0)
            # -4 仓位均值
            mean_weights = df.mean(axis=1)
            # -5 仓位标准差
            std_weights = df.std(axis=1)
            # -6 换手率（相邻两期仓位变化总和的绝对值/2）
            turnover = df.diff().abs().sum(axis=1) / 2
            # -7 交易费率
            fee_ratio = 2 * commission_rate + stamp_tax_rate + 2 * transfer_rate + 2 * slippage_rate
            transaction_fee_rate = turnover * fee_ratio
            transaction_fee_rate.iloc[0] = commission_rate + transfer_rate + slippage_rate

            stats_df = pd.DataFrame({
                "日期": df.index,
                "持股数量": hold_counts,
                "最大仓位": max_weights,
                "最小仓位": min_weights,
                "仓位均值": mean_weights,
                "标准差": std_weights,
                "换手率": turnover,
                "交易费率": transaction_fee_rate
            })
            stats_df["group"] = group_name
            all_stats.append(stats_df)

        daily_stats = pd.concat(all_stats, ignore_index=True)
        daily_stats['日期'] = pd.to_datetime(daily_stats['日期'])

        # -2 根据周期参数进行聚合
        if cycle == "ALL":
            grouped = daily_stats.groupby("group")
            agg_stats = grouped.agg({
                "持股数量": "mean",
                "最大仓位": "max",
                "最小仓位": "min",
                "仓位均值": "mean",
                "标准差": "mean",
                "换手率": "sum",
                "交易费率": "sum"
            }).reset_index()
            agg_stats["日期"] = "ALL"
        elif cycle in ("M", "Y"):
            grouped = daily_stats.groupby([
                "group",
                pd.Grouper(key="日期", freq=cycle)
            ])
            agg_stats = grouped.agg({
                "持股数量": "mean",
                "最大仓位": "max",
                "最小仓位": "min",
                "仓位均值": "mean",
                "标准差": "mean",
                "换手率": "sum",
                "交易费率": "sum"
            }).reset_index()
        else:
            # 默认返回日级别数据
            return daily_stats

        # -3 整理并返回结果
        # 调整列顺序以保持一致性
        column_order = ["日期", "group", "持股数量", "最大仓位", "最小仓位",
                        "仓位均值", "标准差", "换手率", "交易费率"]
        return agg_stats[column_order]

    def _calc_ic_stats(
            self,
            grouped_data: dict[str, pd.DataFrame],
            ic_test: bool = False
    ) -> dict:
        """
        模型结果IC统计
        :param grouped_data: 模型结果
        :param ic_test: 是否进行IC评估
        """
        return self.calc_model_ic_metrics(
                grouped_data, "综合Z值", self.cycle
            ) if ic_test else {}

    def _calc_model_metrics(
            self,
            grouped_data: dict[str, pd.DataFrame],
            transaction_fee_rate: pd.DataFrame
    ) -> dict:
        """
        模型结果收益率统计
        :param grouped_data: 模型结果
        :param transaction_fee_rate: 交易费率
        """
        return {
            # 交易费率（换手率）
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                prefix="换手率估计", fixed_cost=False, transaction_fee_rate=transaction_fee_rate
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="mv_weight", prefix="换手率估计_mw",
                fixed_cost=False, transaction_fee_rate=transaction_fee_rate
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="position_weight", prefix="换手率估计_pw",
                fixed_cost=False, transaction_fee_rate=transaction_fee_rate
            ),
            # 交易费率（固定）
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                prefix="0.0"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="mv_weight", prefix="0.0_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="position_weight", prefix="0.0_pw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                trade_cost=0.01, prefix="0.01"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="mv_weight", trade_cost=0.01, prefix="0.01_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="position_weight", trade_cost=0.01, prefix="0.01_pw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                trade_cost=0.03, prefix="0.03"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="mv_weight", trade_cost=0.03, prefix="0.03_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="position_weight", trade_cost=0.03, prefix="0.03_pw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                trade_cost=0.05, prefix="0.05"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="mv_weight", trade_cost=0.05, prefix="0.05_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.model_setting.group_label,
                mode="position_weight", trade_cost=0.05, prefix="0.05_pw"
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

    def _store_to_excel(
            self,
            model_data: pd.DataFrame,
            file_name: str
    ) -> None:
        """存储模型分组描述性统计"""
        DataStorage(self.storage_dir).write_df_to_excel(
            model_data,
            file_name=file_name,
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
    # 公开 API 方法
    # --------------------------
    def run(self) -> None:
        """执行完整分析流程"""
        self.logger.info(f"start: {str(self.model)} | {self.filter_mode}")

        # 存储模型信息
        self._store_model_setting()

        # 执行分析
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
            model_setting=self.model_setting,
            descriptive_factors=self.DESCRIPTIVE_FACTOR
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
        # 交易统计
        trading_stats = self._calc_trading_stats(model_df)
        trading_stats_month = self._calc_trading_stats(model_df, cycle="M")
        trading_stats_year = self._calc_trading_stats(model_df, cycle="Y")
        trading_stats_all = self._calc_trading_stats(model_df, cycle="ALL")
        # 描述性统计
        descriptive = self._calc_descriptive_factors(model_df)
        descriptive_month = self._calc_descriptive_factors(model_df, "M")
        descriptive_year = self._calc_descriptive_factors(model_df, "Y")
        descriptive_all = self._calc_descriptive_factors(model_df, "ALL")
        # IC/收益率/评估/覆盖度
        result = {
            **self._calc_ic_stats(
                model_data,
                ic_test=True if "综合Z值" in model_df.columns else False
            ),
            **self._calc_model_metrics(
                model_data,
                trading_stats.groupby("日期").apply(lambda x: x.set_index("group")["交易费率"])
            ),
            **{"模型评估指标": metrics_df},
            "coverage": self.calc_coverage(model_data, self.listed_nums),
        }

        # ---------------------------------------
        # 存储、可视化
        # ---------------------------------------
        self.logger.info("---------- 结果存储、可视化 ----------")
        # IC/收益率
        self._draw_charts(self.storage_dir, result, self.visual_setting)
        # 每日持仓
        self._store_grouped_data(model_data)
        # 描述性统计
        self._store_to_excel(descriptive, "描述性统计")
        self._store_to_excel(descriptive_month, "月度描述性统计")
        self._store_to_excel(descriptive_year, "年度描述性统计")
        self._store_to_excel(descriptive_all, "总描述性统计")
        # 交易统计
        self._store_to_excel(trading_stats, "交易统计")
        self._store_to_excel(trading_stats_month, "月度交易统计")
        self._store_to_excel(trading_stats_year, "年度交易统计")
        self._store_to_excel(trading_stats_all, "总交易统计")
        # self._store_selected_factors(selected_factors)
