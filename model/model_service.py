"""
模型分析业务层
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml

from base_service import BaseService
from range_filter import RangeFilter
from constant.path_config import DataPATH
from constant.quant import ModelVisualization
from constant.type_ import CYCLE, validate_literal_params
from model.dimensionality_reduction import FactorCollinearityProcessor
from data_storage import DataStorage


import warnings
warnings.filterwarnings("ignore")


#####################################################
class ModelAnalyzer(BaseService):
    """模型分析"""

    CORE_FACTOR = [
        "对数市值", "open", "close", "pctChg"
    ]
    DESCRIPTIVE_FACTOR = [
        "市值", "市净率"
    ]

    def __init__(
            self,
            model: callable,
            model_setting: dataclass,
            source_dir: str | Path,
            storage_dir: str | Path,
            cycle: CYCLE,
            benchmark_code: str = "000300"
    ):
        """
        :param model_setting: 模型配置
        :param source_dir: 原始数据目录名称
        :param storage_dir: 存储数据目录名称
        :param cycle: 周期
        :param benchmark_code: 基准指数代码
        """
        self.storage_dir = storage_dir
        self.source_dir = source_dir
        self.model = model
        self.model_setting = model_setting
        self.cycle = cycle
        self.benchmark_code = benchmark_code
        self.filter_mode = self.model_setting.filter_mode

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
        self.storage_dir = DataPATH.QUANT_MODEL_ANALYSIS_RESULT / self.storage_dir
        self.source_dir = DataPATH.QUANT_CONVERT_RESULT / self.source_dir

        # --------------------------
        # 数据
        # --------------------------
        self.industry_mapping = self.load_industry_mapping()
        self.listed_nums = self.load_listed_nums()
        self.raw_data = self.load_factor_data(self.source_dir)
        self.target_codes = self._get_target_codes()

        # --------------------------
        # 因子名
        # --------------------------
        self.support_factors_name = self._get_support_factors()
        self.factors_name = self._get_factors_name()
        self.denominators_name = self._get_denominators_name()
        self.valid_factors_name = self._get_valid_factor()

        # --------------------------
        # 其他参数
        # --------------------------
        self.visual_setting = ModelVisualization()
        self.group_label = self.setup_group_label(self.model_setting.group_nums)

    # --------------------------
    # 数据类 方法
    # --------------------------
    def _get_support_factors(
            self
    ) -> list[str]:
        """获取分析因子之外所需的因子"""
        factors = (
                self.DESCRIPTIVE_FACTOR
                + self.CORE_FACTOR
                + self.filter.PARAMETERS
        )
        return [f for f in set(factors) if f]

    def _get_factors_name(
            self
    ) -> list[str]:
        """获取因子名"""
        return [
            setting.factor_name
            for setting in self.model_setting.factors_setting
        ]

    def _get_denominators_name(
            self
    ) -> list[str]:
        """获取重构因子所需的分母"""
        return [
            setting.restructure_denominator
            for setting in self.model_setting.factors_setting
            if setting.restructure
        ]

    def _get_valid_factor(
            self
    ) -> list[str]:
        """获取全部所需的因子"""
        valid_factor = list(set(
            self.factors_name
            + self.support_factors_name
            + self.denominators_name
        ))
        # 排序（支持复现）
        valid_factor.sort()
        return valid_factor

    def _get_target_codes(
            self
    ) -> list[str]:
        """获取目标代码列表"""
        target_codes = self.loader.get_industry_codes(
            sheet_name="Total_A",
            industry_info=self.model_setting.industry_info,
            return_type="list"
        )
        target_codes.sort()
        if target_codes:
            return target_codes
        else:
            raise ValueError(f"目标代码为空")

    # --------------------------
    # 数据处理
    # --------------------------
    def _data_process(
            self,
            raw_data: dict[str, pd.DataFrame],
            valid_factors: list[str],
            factors_setting: list[dataclass]
    ) -> dict[str, pd.DataFrame]:
        """
        数据处理
            -1 截取（全部因子） -2 平移（除 pctChg 之外） -3 过滤（过滤因子）
            -4 预处理（回测因子） -5 分组（预处理因子、使用过去的因子给当下分组，无未来函数）
        :param raw_data: 原始数据
        :param valid_factors: 所需全部因子
        :param factors_setting: 因子配置
        :return: 处理后的数据
        """
        valid_factor_data = self.valid_data_filter(raw_data, valid_factors)

        add_industry_data = self.add_industry(
            valid_factor_data,
            self.industry_mapping,
            self.model_setting.class_level
        )

        shifted_data = self.processor.shift_factors(
            add_industry_data, self.model_setting.lag_period
        )

        filtered_data = self.filter(
            data=shifted_data,
            filter_mode=self.filter_mode,
            cycle=self.cycle
        ).run()

        processed_data = self.processor.preprocessing_factors_by_setting(
            filtered_data,
            factors_setting
        )

        valid_codes_data = self._get_valid_codes(
            processed_data
        )

        return valid_codes_data

    def _get_valid_codes(
            self,
            raw_data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """过滤所需代码，且行数需大于等于分组数"""
        result = {}
        for date, df in raw_data.items():
            index_set = set(df.index)
            valid_codes = [code for code in self.target_codes if code in index_set]
            if not valid_codes:
                continue

            # 提取数据并检查行数
            filtered_df = df.loc[valid_codes]
            if filtered_df.shape[0] >= self.model_setting.group_nums:
                result[date] = filtered_df

        return result

    # --------------------------
    # 计算指标
    # --------------------------
    def _calc_model_metrics(
            self,
            grouped_data: dict[str, pd.DataFrame]
    ) -> dict:
        """计算模型指标"""
        ic_stats = self.calc_ic_metrics(
            grouped_data, "综合Z值", self.cycle
        )
        ic_mean = ic_stats["ic_stats"].loc["ic", "ic_mean"]
        reverse = True if ic_mean < 0 else False

        return {
            **{
                "coverage": self.calc_coverage(grouped_data, self.listed_nums),
                "desc_stats": self.get_desc_stats(
                    grouped_data,
                    list(set(self.factors_name + self.DESCRIPTIVE_FACTOR))
                )
            },
            **ic_stats,
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                reverse=reverse, prefix="0.0"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                mode="mv_weight", reverse=reverse, prefix="0.0_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                mode="position_weight", reverse=reverse, prefix="0.0_pw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                reverse=reverse, trade_cost=0.01, prefix="0.01"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                mode="mv_weight", reverse=reverse, trade_cost=0.01, prefix="0.01_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                mode="position_weight", reverse=reverse, trade_cost=0.01, prefix="0.01_pw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                reverse=reverse, trade_cost=0.03, prefix="0.03"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                mode="mv_weight", reverse=reverse, trade_cost=0.03, prefix="0.03_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                mode="position_weight", reverse=reverse, trade_cost=0.03, prefix="0.03_pw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                reverse=reverse, trade_cost=0.05, prefix="0.05"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
                mode="mv_weight", reverse=reverse, trade_cost=0.05, prefix="0.05_mw"
            ),
            **self.calc_return_metrics(
                grouped_data, self.cycle, self.group_label,
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
        # ---------------------------------------
        # 数据处理
        # ---------------------------------------
        processed_data = self._data_process(
            self.raw_data,
            self.valid_factors_name,
            self.model_setting.factors_setting,
        )
        if not processed_data:
            raise ValueError("过滤数据为空值")

        processed_factors_name = [
            f"processed_{factor_name}"
            for factor_name in self.factors_name
        ]

        # ---------------------------------------
        # 因子降维
        # ---------------------------------------
        # from model.dimensionality_reduction import FactorPCA
        # pca = FactorPCA(n_components=0.95)
        # # 执行降维
        # reduced_data = pca.fit_transform(processed_data["2025-02-28"][processed_factors_name])
        #
        # print("\n降维结果:")
        # print(reduced_data.head())
        #
        # print("\n主成分因子载荷:")
        # print(pca.get_components())
        #
        # print("\n方差解释:")
        # print(pca.get_variance())
        # print(dd)
        print(processed_data.keys())
        # ---------------------------------------
        # 因子去多重共线性（vif + 对称正交）
        # ---------------------------------------
        collinearity = FactorCollinearityProcessor(processed_factors_name)
        selected_factors = collinearity.fit_transform(processed_data)
        if self.model_setting.orthogonal:
            processed_data = self.processor.calc_symmetric_orthogonal(processed_data, selected_factors)

        # ---------------------------------------
        # 预拟合
        # ---------------------------------------
        beta_feature = self.evaluate.test.calc_beta_feature(
            processed_data, processed_factors_name, "pctChg"
        )
        r_squared = self.evaluate.test.calc_r_squared(
            processed_data, processed_factors_name, "pctChg"
        )

        # ---------------------------------------
        # 生成模型
        # ---------------------------------------
        model = self.model(
            raw_data=processed_data,
            factors_name=selected_factors,
            group_nums=self.model_setting.group_nums,
            group_label=self.group_label,
            group_mode=self.model_setting.group_mode,
            factor_weight_method=self.model_setting.factor_weight_method,
            factor_weight_window=self.model_setting.factor_weight_window,
            position_weight_method=self.model_setting.position_weight_method,
            position_distribution=self.model_setting.position_distribution
        )
        grouped_data = model.run()
        print(grouped_data.keys())
        # ---------------------------------------
        # 模型评估
        # ---------------------------------------
        result = {
            **self._calc_model_metrics(grouped_data),
            **{
                "beta_feature": beta_feature
            },
            **{
                "r_squared": r_squared
            }
        }

        # ---------------------------------------
        # 存储、可视化
        # ---------------------------------------
        self._draw_charts(self.storage_dir, result, self.visual_setting)
        self._store_grouped_data(grouped_data)
        self._store_selected_factors(selected_factors)

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
