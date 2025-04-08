from pathlib import Path
from loguru import logger

import pandas as pd

from evaluation import Evaluation
from utils.drawer import Drawer
from utils.data_loader import DataLoader
from utils.quant_processor import QuantProcessor
from constant.type_ import CLASS_LEVEL, CYCLE, CALC_RETURN_MODE, validate_literal_params


####################################################
class BaseService:
    """应用基类"""

    loader = DataLoader
    draw = Drawer
    evaluate = Evaluation()
    processor = QuantProcessor

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
    def load_factor_data(
            cls,
            source_dir: Path
    ) -> dict[str, pd.DataFrame]:
        """加载因子数据（分析因子、双重排序因子、基础量价因子、描述性因子、过滤因子）"""
        return dict(sorted(
            {
                f.stem: cls.loader.load_parquet(f)
                for f in source_dir.glob("*.parquet")
                if f.is_file()
            }.items(),
            key=lambda x: x[0]
        ))

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
        """加载行业数据"""
        return cls.loader.load_parquet(path)

    # --------------------------
    # 数据处理
    # --------------------------
    @classmethod
    @validate_literal_params
    def add_industry(
            cls,
            raw_data: dict[str, pd.DataFrame],
            industry_mapping: pd.DataFrame,
            class_level: CLASS_LEVEL,
    ) -> dict[str, pd.DataFrame]:
        """
        加入行业分类数据
        :param raw_data: 原始数据
        :param industry_mapping: 行业映射数据
        :param class_level: 行业分类级数
        :return: 具有行业信息的数据
        """
        industry = (
            industry_mapping[["股票代码", class_level]]
            .set_index("股票代码")
            .rename(columns={
                class_level: "行业"
            })
        )
        return {
            date: df.join(industry, how="left").fillna({"行业": "未知行业"})
            for date, df in raw_data.items()
        }

    @classmethod
    def valid_data_filter(
            cls,
            raw_data: dict[str, pd.DataFrame],
            valid_factors: list[str]
    ) -> dict[str, pd.DataFrame]:
        """
        有效数据过滤
        :param raw_data: 原始数据
        :param valid_factors: 有效因子
        :return: 过滤后的数据
        """
        required_col = set(valid_factors)

        result = {}
        for date, raw_df in raw_data.items():
            missing_col = required_col - set(raw_df.columns)
            if missing_col:
                print(f"{date} | 缺少以下列: {', '.join(missing_col)}")
            else:
                valid_data = raw_df[valid_factors].dropna(how="any")
                if not valid_data.empty:
                    result[date] = valid_data

        return result

        # return {
        #     date: cleaned_df
        #     for date, raw_df in raw_data.items()
        #     # 应有所需的全部数据
        #     if required_col.issubset(raw_df.columns)
        #     # 删除任意因子缺失值所在的行
        #     if not (
        #         cleaned_df := raw_df[valid_factors].dropna(subset=valid_factors, how="any")
        #     ).empty
        # }

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
                    pd.Series({"duration": f"{round(duration, 2)}Y"}),
                    pd.Series({"corr": cls.evaluate.test.rank_corr_test(annualized_return[group_label])}),
                    cls.evaluate.test.jonckheere_terpstra_test(grouped_return[group_label]),
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
