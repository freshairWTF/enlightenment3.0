from pathlib import Path
from loguru import logger

import pandas as pd

from evaluation import Evaluation
from utils.drawer import Drawer
from utils.loader import DataLoader
from constant.type_ import CLASS_LEVEL, CYCLE, CALC_RETURN_MODE, validate_literal_params
from analysis.kline_metrics import KLineMetrics
from constant.path_config import DataPATH


####################################################
class QuantService:
    """量化基类"""

    loader = DataLoader
    draw = Drawer
    evaluate = Evaluation()

    # --------------------------
    # 初始化
    # --------------------------
    @classmethod
    def setup_group_label(
            cls,
            group_nums: int
    ) -> list[str]:
        """设置分组标签"""
        return list(str(i) for i in range(0, 100, int(100 / group_nums)))[: group_nums]

    # --------------------------
    # 数据加载
    # --------------------------
    @classmethod
    def load_factors_value(
            cls,
            source_dir: Path
    ) -> pd.DataFrame:
        """加载因子数据并合并为单个DataFrame，日期作为新列"""
        # 加载原始字典数据
        data_dict = dict(sorted(
            {
                f.stem: cls.loader.load_parquet(f)
                for f in source_dir.glob("*.parquet")
                if f.is_file()
            }.items(),
            key=lambda x: x[0]
        ))

        # 创建包含日期列的DataFrame列表
        dfs_with_date = []
        for date_str, df in data_dict.items():
            # 添加日期列 [5](@ref)
            df = df.copy()  # 避免修改原始DataFrame
            df['date'] = date_str  # 添加日期列
            dfs_with_date.append(df)

        # 合并所有DataFrame
        combined_df = pd.concat(dfs_with_date)
        # 代码索引成列
        combined_df = combined_df.reset_index().rename(columns={'index': '股票代码'})
        # 转换日期
        combined_df['date'] = pd.to_datetime(combined_df['date'], format='%Y-%m-%d').dt.date

        return combined_df

    @classmethod
    def load_industry_mapping(
            cls
    ) -> pd.DataFrame:
        """加载行业映射表"""
        return (
            cls.loader.get_industry_codes(
                sheet_name="Total_A",
                industry_info={"全部": "一级行业"}
            ).assign(**{"股票代码": lambda df: df["股票代码"].str.split(".").str[0]})
        )

    @classmethod
    def load_listed_nums(
            cls
    ) -> pd.Series:
        """加载上市数"""
        return cls.loader.get_listed_nums()["listed_nums"]

    @classmethod
    def load_index_kline(
            cls,
            path: Path | str
    ) -> pd.DataFrame:
        """加载指数k线数据"""
        df = cls.loader.load_parquet(path).set_index("date")
        df.index = pd.to_datetime(df.index)
        return df

    @classmethod
    @validate_literal_params
    def _calculate_kline(
            cls,
            kline: pd.DataFrame,
            cycle: CYCLE,
            methods: dict[str, list[float]]
    ) -> pd.DataFrame:
        """计算量价指标"""
        calculator = KLineMetrics(
            kline_data=kline,
            cycle=cycle,
            methods=methods,
            function_map=cls.loader.load_yaml_file(DataPATH.KLINE_METRICS, "FUNCTION_OF_KLINE")
        )
        calculator.calculate()
        return calculator.metrics.round(4)

    # --------------------------
    # 数据处理
    # --------------------------
    @classmethod
    @validate_literal_params
    def add_industry(
            cls,
            raw_data: pd.DataFrame,
            industry_mapping: pd.DataFrame,
            class_level: CLASS_LEVEL,
    ) -> pd.DataFrame:
        """
        加入行业分类数据
        :param raw_data: 原始数据
        :param industry_mapping: 行业映射数据
        :param class_level: 行业分类级数
        :return: 具有行业信息的数据
        """
        industry = (
            industry_mapping[["股票代码", class_level]]
            .rename(columns={
                class_level: "行业"
            })
        )
        return pd.merge(raw_data, industry, how="left", on="股票代码")

    @classmethod
    def valid_factors_filter(
            cls,
            raw_data: pd.DataFrame,
            valid_factors: list[str],
            date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        有效数据过滤
        :param raw_data: 原始数据
        :param valid_factors: 有效因子
        :param date_col: 标识日期的列名，默认为 date
        :return: 过滤后的数据
        """
        # 确保日期列存在
        if date_col not in raw_data.columns:
            raise ValueError(f"DataFrame中缺少日期列: {date_col}")

        # 获取所有唯一日期
        unique_dates = raw_data[date_col].unique()
        result_dfs = []                             # 存储每个日期处理后的DataFrame

        # 检查因子列是否存在
        missing_factors = set(valid_factors) - set(raw_data.columns)
        if missing_factors:
            print(f"警告：全局缺少以下因子列: {', '.join(missing_factors)}")

        for date in unique_dates:
            # 获取当前日期的数据子集
            date_mask = raw_data[date_col] == date
            date_df = raw_data.loc[date_mask]

            # 列检查 检查当前日期是否缺少因子列
            valid_df = date_df[valid_factors].dropna(how="all", axis=1)
            date_missing = set(valid_factors) - set(valid_df.columns)
            if date_missing:
                print(f"{date} | 缺少以下列: {', '.join(date_missing)}")
                continue

            # 行检查 筛选有效因子并删除缺失值
            valid_df = date_df[valid_factors].dropna(how="any")
            if valid_df.empty:
                print(f"{date} | 缺少行")
            else:
                result_dfs.append(valid_df)

        # 合并所有有效数据
        return pd.concat(result_dfs, ignore_index=False) if result_dfs else pd.DataFrame()

    # --------------------------
    # 指标计算
    # --------------------------
    @classmethod
    def calc_coverage(
            cls,
            data: dict[str, pd.DataFrame],
            listed_nums: pd.Series
    ) -> pd.DataFrame:
        """数据覆盖度"""
        return pd.DataFrame.from_dict(
            {
                date: df.shape[0] / listed_nums[date[: 7]][0] * 100
                for date, df in data.items()
                if date[: 7] in listed_nums.index
            },
            orient="index",
            columns=["覆盖度"]
        )

    @classmethod
    def get_desc_stats(
            cls,
            data: dict[str, pd.DataFrame],
            desc_factors: list[str]
    ) -> pd.DataFrame:
        """描述性统计"""
        desc = pd.concat(
            [
                df.groupby("group")[desc_factors].mean()
                for date, df in data.items()
            ],
        ).groupby(level=0).mean()

        desc["市值"] /= 10 ** 8

        return desc.T

    @classmethod
    @validate_literal_params
    def calc_ic_metrics(
            cls,
            grouped_data: dict[str, pd.DataFrame],
            factor_col: str,
            cycle: CYCLE
    ) -> dict[str, pd.DataFrame | pd.Series]:
        """
        计算ic指标
        :param grouped_data: 分组数据
        :param factor_col: 因子列名
        :param cycle: 周期
        :return: ic指标群
        """
        ic_df = cls.evaluate.ic.calc_ic(
            grouped_data, factor_col, "pctChg"
        )
        ic_decay = cls.evaluate.ic.calc_ic_decay(
            grouped_data, factor_col
        )
        ic_mean = ic_df.mean()

        return {
            "ic": ic_df,
            "ic_decay": ic_decay,
            "ic_cumsum": ic_df.cumsum(),
            "ic_stats": pd.DataFrame.from_dict(
                dict(
                    ic_mean=ic_mean,
                    ic_significance=cls.evaluate.ic.calc_ic_significance(ic_df),
                    ic_winning_rate=cls.evaluate.returns.winning_rate(
                        ic_df, False if ic_mean.loc["ic"] >= 0 else True
                    ),
                    ic_ir=cls.evaluate.ic.calc_icir(ic_df, cycle),
                    half_life=cls.evaluate.ic.get_half_life(ic_mean.loc["ic"], ic_decay["ic"])
                )
            )
        }

    @classmethod
    @validate_literal_params
    def calc_model_ic_metrics(
            cls,
            grouped_data: dict[str, pd.DataFrame],
            factor_col: str,
            cycle: CYCLE
    ) -> dict[str, pd.DataFrame | pd.Series]:
        """
        计算ic指标
        :param grouped_data: 分组数据
        :param factor_col: 因子列名
        :param cycle: 周期
        :return: ic指标群
        """
        ic_df = cls.evaluate.ic.calc_ic(
            grouped_data, factor_col, "pctChg"
        )
        ic_mean = ic_df.mean()

        return {
            "ic": ic_df,
            "ic_cumsum": ic_df.cumsum(),
            "ic_stats": pd.DataFrame.from_dict(
                dict(
                    ic_mean=ic_mean,
                    ic_significance=cls.evaluate.ic.calc_ic_significance(ic_df),
                    ic_winning_rate=cls.evaluate.returns.winning_rate(
                        ic_df, False if ic_mean.loc["ic"] >= 0 else True
                    ),
                    ic_ir=cls.evaluate.ic.calc_icir(ic_df, cycle)
                )
            )
        }

    @classmethod
    @validate_literal_params
    def calc_return_metrics(
            cls,
            grouped_data: dict[str, pd.DataFrame],
            cycle: CYCLE,
            group_label: list[str],
            mode: CALC_RETURN_MODE = "equal",
            reverse: bool = False,
            trade_cost: float = 0.0,
            prefix: str = ""
    ) -> dict[str, pd.DataFrame | pd.Series]:
        """
        计算收益率、比率指标
        :param grouped_data: 分组数据
        :param cycle: 周期
        :param group_label: 分组标签
        :param mode: 收益率计算模式
        :param reverse: 多空反转
        :param trade_cost: 手续费率
        :param prefix: 前缀
        :return: 收益率、比率指标
        """
        # 最劣/最优 标签
        min_label, max_label = group_label[0], group_label[-1]

        # --------------------------
        # 指标计算
        # --------------------------
        grouped_return = cls.evaluate.returns.calc_group_returns(
            grouped_data,
            cycle,
            max_label,
            min_label,
            mode,
            reverse,
            trade_cost
        )

        cum_return = cls.evaluate.returns.cum_return(grouped_return)
        wtl_ratio = cls.evaluate.returns.win_to_loss_ratio(grouped_return)
        winning_rate = cls.evaluate.returns.winning_rate(grouped_return)
        maximum_drawdown = cls.evaluate.returns.maximum_drawdown(cum_return)
        annualized_return = cls.evaluate.returns.annualized_return(grouped_return, cycle)
        duration = cls.evaluate.returns.calc_duration(grouped_return, cycle)

        result = {
            "returns": grouped_return,
            "cum_returns": cum_return,
            "returns_stats": pd.DataFrame.from_dict(
                {
                    "annualized_return": annualized_return,
                    "t_value": cls.evaluate.returns.t_value(grouped_return),
                    "winning_rate": winning_rate,
                    "wtl_ratio": wtl_ratio,
                    "maximum_drawdown": maximum_drawdown,
                    "maximum_drawdown_period": cls.evaluate.returns.maximum_drawdown_period(grouped_return),
                    "sharp_ratio": cls.evaluate.ratio.sharpe_ratio(grouped_return, cycle),
                    "sortino_ratio": cls.evaluate.ratio.sortino_ratio(grouped_return, cycle),
                    "kelly_ratio": cls.evaluate.ratio.kelly_ratio(winning_rate, wtl_ratio),
                    "sterling_ratio": cls.evaluate.ratio.sterling_ratio(grouped_return, maximum_drawdown, cycle),
                    "1Y_best_scenario": cls.evaluate.returns.scenario_return(grouped_return, cycle),
                    "1Y_worst_scenario": cls.evaluate.returns.scenario_return(grouped_return, cycle, False),
                    "1Y_highest_return": cls.evaluate.returns.highest_rolling_return(grouped_return, cycle, 1),
                    "1Y_lowest_return": cls.evaluate.returns.lowest_rolling_return(grouped_return, cycle, 1),
                    "3Y_highest_return": cls.evaluate.returns.highest_rolling_return(grouped_return, cycle, 3),
                    "3Y_lowest_return": cls.evaluate.returns.lowest_rolling_return(grouped_return, cycle, 3),
                    "5Y_highest_return": cls.evaluate.returns.highest_rolling_return(grouped_return, cycle, 5),
                    "5Y_lowest_return": cls.evaluate.returns.lowest_rolling_return(grouped_return, cycle, 5),
                    "7Y_highest_return": cls.evaluate.returns.highest_rolling_return(grouped_return, cycle, 7),
                    "7Y_lowest_return": cls.evaluate.returns.lowest_rolling_return(grouped_return, cycle, 7),
                    "10Y_highest_return": cls.evaluate.returns.highest_rolling_return(grouped_return, cycle, 10),
                    "10Y_lowest_return": cls.evaluate.returns.lowest_rolling_return(grouped_return, cycle, 10),
                }
            ).T,
            "basic_stats": pd.concat(
                [
                    pd.Series(
                        {"duration": f"{round(duration, 2)}Y"}
                    ),
                    pd.Series(
                        {"corr": cls.evaluate.test.rank_corr_test(annualized_return[group_label])}
                    ),
                    cls.evaluate.test.jonckheere_terpstra_test(grouped_return[group_label]),
                    pd.Series(
                        {
                            "j_shape_p_value": cls.evaluate.returns.check_j_shape_feature(
                                grouped_data,
                                (group_label[0], group_label[1]) if reverse else
                                (group_label[-1], group_label[-2])
                            )
                        }
                    ),
                ]
            ).to_frame("value").T,
        }

        return {f"{prefix}_{k}": v for k, v in result.items()} if prefix else result

    # --------------------------
    # 存储、可视化 方法
    # --------------------------
    @classmethod
    def _draw_charts(
            cls,
            path: Path,
            data: dict[str, pd.DataFrame | pd.Series | float],
            visual_setting
    ) -> None:
        """
        生成可视化图表
        :param path: 存储路径
        :param data: 可视化数据
        :param visual_setting: 可视化设置
        """
        cls.drawer = Drawer(
            path=path,
            pages_name=visual_setting.pages_name,
            pages_config=visual_setting.pages_config,
            data_container=data
        )
        cls.drawer.draw()
        cls.drawer.render()

    # --------------------------
    # 其他
    # --------------------------
    @staticmethod
    def setup_logger(
            path: Path
    ) -> logger:
        """配置日志记录器"""
        logger.add(path / "日志.log", rotation="1 MB", level="INFO")
        return logger
