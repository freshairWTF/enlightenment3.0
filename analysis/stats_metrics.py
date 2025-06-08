import pandas as pd

from base_metrics import Metrics
from constant.type_ import CYCLE, validate_literal_params
from utils.processor import DataProcessor


###############################################################
class StatisticsMetrics(Metrics):
    """
    统一统计类，支持汇总数据和个股分析
    """

    processor = DataProcessor()

    @validate_literal_params
    def __init__(
            self,
            data_container,
            cycle: CYCLE,
            financial_cycle: CYCLE,
            methods: list[str],
            function_map: dict[str, str],
            index_kline: pd.DataFrame | None = None
    ):
        """
        :param data_container: 数据源格式
                            {
                            "financial": {
                                indicator: financial_df
                                }
                            ......
                          }
        :param cycle: 周期
        :param financial_cycle: 周期 -1 非量化 财务是financial_cycle 估值是cycle
        :param methods: 需要实现的方法
        :param function_map: 已定义的方法对应方法名
        :param index_kline: 指数k线
        """
        self.data_container = data_container
        self.function_map = function_map
        self.cycle = cycle
        self.financial_cycle = financial_cycle
        self.methods = methods
        self.index_kline = index_kline

        # 获取窗口数
        self.window = self._setup_window(self.cycle)
        self.financial_window = self._setup_window(self.financial_cycle)

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def calculate(
            self
    ) -> None:
        """计算接口"""
        for method in self.methods:
            if method in self.function_map:
                method_name = self.function_map.get(method)
                if method_name and hasattr(self, method_name):
                    getattr(self, method_name)()
                else:
                    raise ValueError(f"未实现的方法: {method}")
            else:
                raise ValueError(f"未定义的指标: {method}")

    # --------------------------
    # 数据特征
    # --------------------------
    def _yoy(
            self
    ) -> None:
        """同比计算"""
        applicable = {
            "financial": self.financial_window,
            "rolling_financial": self.financial_window,
        }
        suffix = "_yoy"

        for key, window in applicable.items():
            data = self.data_container.get(key)
            if data:
                self.data_container[key].update(
                    {
                        f"{k}{suffix}": self._calc_growth_rate(
                            data=df[df.columns[df.columns != "公司简称"]],
                            window=window
                        ) for k, df in data.items() if "_"not in k
                    }
                )

    def _qoq(
            self
    ) -> None:
        """环比计算"""
        applicable = {
            "financial": 1,
            "rolling_financial": 1,
        }
        suffix = "_qoq"

        for key, window in applicable.items():
            data = self.data_container.get(key)
            if data:
                self.data_container[key].update(
                    {
                        f"{k}{suffix}": self._calc_growth_rate(
                            data=df[df.columns[df.columns != "公司简称"]],
                            window=window
                        ) for k, df in data.items() if "_"not in k
                    }
                )

    def _cagr(
            self
    ) -> None:
        """复合增速计算"""
        applicable = {
            "financial": None,
            "rolling_financial": None,
        }
        suffix = "_cagr"

        for key, _ in applicable.items():
            data = self.data_container.get(key)
            if data:
                self.data_container[key].update(
                    {
                        f"{k}{suffix}": self._calc_growth_rate(
                            data=df[df.columns[df.columns != "公司简称"]],
                            compound=True
                        ) for k, df in data.items() if "_"not in k
                    }
                )

    def _mean(
            self
    ) -> None:
        """均值计算"""
        applicable = {
            "financial": self.financial_window * 3,
            "rolling_financial": self.financial_window * 3,
            "valuation": self.window * 3
        }
        suffix = "_mean"

        for key, window in applicable.items():
            data = self.data_container.get(key)
            if data:
                self.data_container[key].update(
                    {
                        f"{k}{suffix}": self._calc_rolling(
                            data=df[df.columns[df.columns != "公司简称"]],
                            window=window,
                            min_periods=1,
                            method="mean"
                        ) for k, df in data.items() if "_"not in k
                    }
                )

    def _median(
            self
    ) -> None:
        """中位数计算"""
        applicable = {
            "financial": self.financial_window * 3,
            "rolling_financial": self.financial_window * 3,
            "valuation": self.window * 3,
        }
        suffix = "_median"

        for key, window in applicable.items():
            data = self.data_container.get(key)
            if data:
                self.data_container[key].update(
                    {
                        f"{k}{suffix}": self._calc_rolling(
                            data=df[df.columns[df.columns != "公司简称"]],
                            window=window,
                            min_periods=1,
                            method="median"
                        ) for k, df in data.items() if "_"not in k
                    }
                )

    def _normalized(
            self
    ) -> None:
        """归一化计算"""
        applicable = {
            "financial": self.financial_window * 3,
            "rolling_financial": self.financial_window * 3,
            "valuation": self.window * 3,
        }
        suffix = "_normalized"

        for key, _ in applicable.items():
            data = self.data_container.get(key)
            if data:
                self.data_container[key].update(
                    {
                        f"{k}{suffix}": self.processor.dimensionless.normalization(
                            factor_values=df[df.columns[df.columns != "公司简称"]]
                        ) * 100 for k, df in data.items() if "_"not in k
                    }
                )

    def _rolling_normalized(
            self
    ) -> None:
        """滚动归一化计算"""
        applicable = {
            "financial": self.financial_window * 3,
            "rolling_financial": self.financial_window * 3,
            "valuation": self.window * 3,
            "kline": self.window * 3
        }
        suffix = "_rolling_normalized"

        for key, window in applicable.items():
            data = self.data_container.get(key)
            if data:
                self.data_container[key].update(
                    {
                        f"{k}{suffix}": self.processor.dimensionless.normalization(
                            factor_values=df[df.columns[df.columns != "公司简称"]],
                            window=window,
                            min_periods=1
                        ) * 100 for k, df in data.items() if "_"not in k
                    }
                )

    # --------------------------
    # 比较
    # --------------------------
    def _proportion(
            self
    ) -> None:
        """占比计算"""
        applicable = {
            "financial": None,
            "rolling_financial": None,
            "valuation": None,
        }
        suffix = "_proportion"

        for key, _ in applicable.items():
            data = self.data_container.get(key)
            if data:
                self.data_container[key].update(
                    {
                        f"{k}{suffix}": self._calc_proportion(
                            data=df[df.columns[df.columns != "公司简称"]]
                        ) for k, df in data.items()
                    }
                )

    def _periodic_testing(
            self,
    ) -> None:
        """周期性检验（营收增速、毛利率、净利润、资本支出波动率）"""


###############################################################
class IndividualStatisticsMetrics(Metrics):
    """
    统一统计类，支持汇总数据和个股分析
    """

    processor = DataProcessor()

    @validate_literal_params
    def __init__(
            self,
            code: str,
            data_container,
            cycle: CYCLE,
            financial_cycle: CYCLE,
            methods: list[str],
            function_map: dict[str, str],
            index_kline: pd.DataFrame | None = None
    ):
        """
        :param data_container: 数据源格式
                {
                "financial": {
                    indicator: financial_df
                    }
                ......
              }
        -->     {
                "financial": {
                    financial: df
                    }
                ......
              }
        :param cycle: 周期
        :param financial_cycle: 周期 -1 非量化 财务是financial_cycle 估值是cycle
        :param methods: 需要实现的方法
        :param function_map: 已定义的方法对应方法名
        :param index_kline: 指数k线
        """
        self.function_map = function_map
        self.cycle = cycle
        self.financial_cycle = financial_cycle
        self.methods = methods
        self.index_kline = index_kline

        # 获取窗口数
        self.window = self._setup_window(self.cycle)
        self.financial_window = self._setup_window(self.financial_cycle)

        # 初始化存储容器
        self.data_container = self._modify_data_container(code, data_container)
        self.data_container = self._flatten_data_container(data_container)

        # -1 所有方法不可叠加 方法名与后缀相同  -2 非数值数据不参与计算
        self.exclude = dir(self) + ["公司简称"]

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def calculate(
            self
    ) -> None:
        """计算接口"""
        for method in self.methods:
            if method in self.function_map:
                method_name = self.function_map.get(method)
                if method_name and hasattr(self, method_name):
                    getattr(self, method_name)()
                else:
                    raise ValueError(f"未实现的方法: {method}")
            else:
                raise ValueError(f"未定义的指标: {method}")

    # --------------------------
    # 初始化方法
    # --------------------------
    @classmethod
    def _modify_data_container(
            cls,
            code: str,
            data_container: dict[str, dict[str, pd.DataFrame]]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """提取数据（financial，rolling_financial，kline，valuation）"""
        for key in ["financial", "rolling_financial", "kline", "valuation"]:
            data_container[key].update(
                {"": data_container[key].pop(code)}
            )
        return data_container

    @classmethod
    def _flatten_data_container(
            cls,
            data_container: dict[str, dict[str, pd.DataFrame]],
    ) -> dict[str, pd.DataFrame]:
        """数据容器字典合一"""
        def flatten_dict(d, parent_key="", sep=""):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return flatten_dict(data_container)

    # --------------------------
    # 统计数据特征 方法
    # --------------------------
    def _yoy(
            self
    ) -> None:
        """同比计算（存储于同一DataFrame中）"""
        applicable = {
            "financial": self.financial_window,
            "rolling_financial": self.financial_window,
        }
        suffix = "_yoy"

        for key, window in applicable.items():
            data = self.data_container.get(key, pd.DataFrame())
            if not data.empty:
                # 正则表达式过滤条件
                exclude_pattern = f"^(?!.*({'|'.join(self.exclude)})$).*"
                filter_data = data.filter(regex=exclude_pattern)

                # 计算同比数据
                yoy_data = self._calc_growth_rate(
                    data=filter_data,
                    window=window
                ).add_suffix(suffix).round(2)

                # 合并
                self.data_container[key] = pd.merge(
                    data,
                    yoy_data,
                    right_index=True,
                    left_index=True
                )

    def _qoq(
            self
    ) -> None:
        """环比计算（存储于同一DataFrame中）"""
        applicable = {
            "financial": 1,
            "rolling_financial": 1
        }
        suffix = "_qoq"

        for key, window in applicable.items():
            data = self.data_container.get(key, pd.DataFrame())
            if not data.empty:
                # 正则表达式过滤条件
                exclude_pattern = f"^(?!.*({'|'.join(self.exclude)})$).*"
                filter_data = data.filter(regex=exclude_pattern)

                # 计算环比数据
                qoq_data = self._calc_growth_rate(
                    data=filter_data,
                    window=window
                ).add_suffix(suffix).round(2)

                # 合并
                self.data_container[key] = pd.merge(
                    data,
                    qoq_data,
                    right_index=True,
                    left_index=True
                )

    def _cagr(
            self
    ) -> None:
        """复合增速计算（存储于同字典中，键名 key_cagr）"""
        applicable = {
            "financial": None,
            "rolling_financial": None,
        }
        suffix = "_cagr"

        for key, _ in applicable.items():
            data = self.data_container.get(key, pd.DataFrame())
            if not data.empty:
                # 正则表达式过滤条件
                exclude_pattern = f"^(?!.*({'|'.join(self.exclude)})$).*"
                filter_data = data.filter(regex=exclude_pattern)

                growth_data = self._calc_growth_rate(
                    data=filter_data,
                    compound=True
                ).round(2)

                if isinstance(growth_data, pd.Series):
                    growth_df = growth_data.to_frame(name="cagr")
                else:
                    growth_df = growth_data

                self.data_container.update(
                    {
                        f"{key}{suffix}": growth_df
                    }
                )

    def _mean(
            self
    ) -> None:
        """均值计算（存储于同字典中，键名 key_mean）"""
        applicable = {
            "financial": self.financial_window * 3,
            "rolling_financial": self.financial_window * 3,
            "valuation": self.window * 3
        }
        suffix = "_mean"

        for key, window in applicable.items():
            data = self.data_container.get(key, pd.DataFrame())
            if not data.empty:
                # 正则表达式过滤条件
                exclude_pattern = f"^(?!.*({'|'.join(self.exclude)})$).*"
                filter_data = data.filter(regex=exclude_pattern)

                mean_data = self._calc_rolling(
                    data=filter_data,
                    window=window,
                    min_periods=1,
                    method="mean"
                ).round(2)

                if isinstance(mean_data, pd.Series):
                    mean_df = mean_data.to_frame(name="mean")
                else:
                    mean_df = mean_data

                self.data_container.update(
                    {
                        f"{key}{suffix}": mean_df
                    }
                )

    def _median(
            self
    ) -> None:
        """中位数计算（存储于同字典中，键名 key_median）"""
        applicable = {
            "financial": self.financial_window * 3,
            "rolling_financial": self.financial_window * 3,
            "valuation": self.window * 3
        }
        suffix = "_median"

        for key, window in applicable.items():
            data = self.data_container.get(key, pd.DataFrame())
            if not data.empty:
                # 正则表达式过滤条件
                exclude_pattern = f"^(?!.*({'|'.join(self.exclude)})$).*"
                filter_data = data.filter(regex=exclude_pattern)

                median_data = self._calc_rolling(
                    data=filter_data,
                    window=window,
                    min_periods=1,
                    method="median"
                ).round(2)

                if isinstance(median_data, pd.Series):
                    median_df = median_data.to_frame(name="median")
                else:
                    median_df = median_data

                self.data_container.update(
                    {
                        f"{key}{suffix}": median_df
                    }
                )

    def _normalized(
            self
    ) -> None:
        """归一化计算（存储于同字典中，键名 key_normalized）"""
        applicable = {
            "financial": None,
            "rolling_financial": None,
            "valuation": None
        }
        suffix = "_normalized"

        for key, _ in applicable.items():
            data = self.data_container.get(key, pd.DataFrame())
            if not data.empty:
                # 正则表达式过滤条件
                exclude_pattern = f"^(?!.*({'|'.join(self.exclude)})$).*"
                filter_data = data.filter(regex=exclude_pattern)

                # 计算分位数
                normal = (self.processor.dimensionless.normalization(factor_values=filter_data) * 100).round(2)

                # 合并
                self.data_container.update(
                    {
                        f"{key}{suffix}": normal
                    }
                )

    # --------------------------
    # 比较 方法
    # --------------------------
    def _proportion(
            self
    ) -> None:
        """占比计算（存储于同字典中，键名 key_proportion）"""
        applicable = {
            "financial": None,
            "rolling_financial": None,
            "valuation": None
        }
        suffix = "_proportion"

        for key, _ in applicable.items():
            data = self.data_container.get(key, pd.DataFrame())
            if not data.empty:
                self.data_container.update(
                    {
                        f"{key}{suffix}": self._calc_proportion(
                            data=data.filter(regex="^(?!.*" + "|.*".join(self.exclude) + "$)"),
                        ).round(2)
                    }
                )
