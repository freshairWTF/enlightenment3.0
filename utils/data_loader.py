from _operator import or_
from functools import reduce
from pathlib import Path

import yaml
import pandas as pd

from constant.path_config import DataPATH
from constant.type_ import (
    FINANCIAL_CYCLE, KLINE_CYCLE, KLINE_SHEET,
    INDUSTRY_SHEET, CYCLE, INDUSTRY_RETURN_TYPE,
    validate_literal_params
)
from constant.quant import ANNUALIZED_DAYS


ALIGNED_TO_MONTH_END = ["week", "day", "original_day"]


##############################################################
class DataLoader:
    """
    多功能数据加载器，支持财务数据、行情数据、行业分类等数据的加载和预处理
    """

    # --------------------------
    # 类属性
    # --------------------------


    # --------------------------
    # 通用私有方法
    # --------------------------
    @classmethod
    def _validate_path(
            cls,
            path: str | Path
    ) -> Path:
        """路径验证与转换"""
        path = Path(path) if isinstance(path, str) else path
        if not path.exists():
            raise FileNotFoundError(f"路径不存在：{path}")
        return path

    @classmethod
    def _read_sheet(
            cls,
            path: str | Path,
            sheet_name: str | int = 0
    ) -> pd.DataFrame:
        """统一读取Excel工作表（第一列作为索引、转变date为Timestamp）"""
        path = cls._validate_path(path)
        df = pd.read_excel(path, sheet_name=sheet_name, index_col="date", parse_dates=["date"])
        return df

    @classmethod
    def _read_parquet(
            cls,
            path: str | Path
    ) -> pd.DataFrame:
        """统一读取Excel工作表（第一列作为索引、转变date为Timestamp）"""
        path = cls._validate_path(path)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])

        return df.set_index("date")

    @staticmethod
    def _aligned_to_month_end(
            date: pd.Timestamp
    ) -> pd.Timestamp:
        """将日期调整到当月最后一天"""
        return date + pd.tseries.offsets.MonthEnd(0)

    # --------------------------
    # 财务数据处理（私有方法）
    # --------------------------
    @classmethod
    def _cycle_filter(
            cls,
            df: pd.DataFrame,
            cycle: FINANCIAL_CYCLE
    ) -> pd.DataFrame | None:
        """
        基于周期的数据过滤
        :param df: 原始数据
        :param cycle: 分析周期 (year/half/quarter)
        :return: 过滤后的数据
        """
        month_pattern = {
            "year": [12],
            "half": [6, 12],
            "quarter": [3, 6, 9, 12]
        }.get(cycle, None)

        if not month_pattern:
            return df

        # 提取符合月份模式的时间索引
        filtered_index = df.index[df.index.month.isin(month_pattern)]
        df = df.loc[filtered_index]

        # -1 年度直接 return
        if cycle == "year":
            return df
        # -2 半年度的第一行数据是 6月份
        elif cycle == "half":
            june = df.index[df.index.month == 6]
            if june.empty:
                raise IndexError(f"_cycle_filter: Q2 data not found")
            else:
                return df[df.index >= june[0]]
        # -3 季度的第一行数据是 3月份
        elif cycle == "quarter":
            mar = df.index[df.index.month == 3]
            if mar.empty:
                raise IndexError(f"_cycle_filter: Q1 data not found")
            else:
                return df[df.index >= mar[0]]

    @classmethod
    def _transform_to_non_cumulative(
            cls,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将累计值转换为当期值： -1 按年分组； -2 数值型数据相减；
        :param df: 原始累计值数据
        :return: 转换后的当期值数据
        """
        def __transform_to_non_cumulative(group):
            # 复制原始分组数据（避免修改原始数据）
            group_transformed = group.copy()
            # 选择数值列
            numeric_cols = group.select_dtypes(include=["number"]).columns
            # 对数值列进行差分，并用原始值填充第一行
            group_transformed[numeric_cols] = group[numeric_cols].diff().fillna(group[numeric_cols])
            return group_transformed

        # 分组
        grouped = df.groupby(df.index.year, group_keys=False)
        # 数值型数据相减
        non_cumulative_data = grouped.apply(__transform_to_non_cumulative)

        return non_cumulative_data

    @classmethod
    def _complete_inspection(
            cls,
            df: pd.DataFrame,
            required_columns: list[str] | None,
            start_date: pd.Timestamp | None = None,
            end_date: pd.Timestamp | None = None,
            cycle: FINANCIAL_CYCLE | None = None
    ) -> bool:
        """
        数据完整性检查
        :param df: 需要检查的数据框
        :param required_columns: 必须包含的列
        :param start_date: 要求的起始日期
        :param end_date: 要求的结束日期
        :param cycle: 分析周期
        :return: 是否通过检查
        """
        if df.empty:
            print("数据完整性检查失败：空数据框")
            return False

        # 检查必要列
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            print(f"缺失必要数据列：{missing}")
            return False

        # 检查时间范围
        if start_date and df.index[0] > start_date:
            print(f"起始时间不满足：{df.index[0]} > {start_date}")
            return False

        if end_date and df.index[-1] < end_date:
            print(f"结束时间不满足：{df.index[-1]} < {end_date}")
            return False

        # 检查周期完整性
        if cycle:
            freq = {
                "year": "YE",
                "half": "6M",
                "quarter": "QE"
            }[cycle]
            expected_dates = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq=freq
            )
            missing_dates = expected_dates.difference(df.index)
            if not missing_dates.empty:
                print(f"缺失时间点：{missing_dates.tolist()}")
                return False

        return True

    # --------------------------
    # 公开 API 方法
    # --------------------------
    @classmethod
    @validate_literal_params
    def get_financial(
            cls,
            code: str,
            cycle: FINANCIAL_CYCLE,
            start_date: str | None = None,
            end_date: str | None = None,
            required_columns: list[str] | None = None,
            inspection: bool = False
    ) -> pd.DataFrame:
        """
        获取标准化财务数据
        :param code: 企业代码
        :param start_date: 起始日期 (YYYY-MM-DD)
        :param end_date: 结束日期 (YYYY-MM-DD)
        :param cycle: 分析周期
        :param required_columns: 必须包含的财务科目
        :param inspection: 完备性价差
        :return: 处理后的财务数据DataFrame
        """
        # 参数处理
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # 读取原始数据
        bs = cls._read_parquet((DataPATH.FINANCIAL_DATA / f"{code}_bs").with_suffix(".parquet"))
        ps = cls._read_parquet((DataPATH.FINANCIAL_DATA / f"{code}_ps").with_suffix(".parquet"))
        cf = cls._read_parquet((DataPATH.FINANCIAL_DATA / f"{code}_cf").with_suffix(".parquet"))

        # 数据预处理
        # -1 周期过滤
        bs = cls._cycle_filter(bs, cycle)
        ps = cls._cycle_filter(ps, cycle)
        cf = cls._cycle_filter(cf, cycle)

        # -2 数据转换
        ps = cls._transform_to_non_cumulative(ps)
        cf = cls._transform_to_non_cumulative(cf)

        # -3 合并数据
        df = pd.concat([bs, ps, cf], axis=1).sort_index()

        # -4 过滤重复列名
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        # -5 时间过滤
        df = df[df.index >= start_date] if start_date is not None else df
        df = df[df.index <= end_date] if end_date is not None else df

        # -6 关键词过滤
        df = df[required_columns] if required_columns else df

        # -7 数据完整性检查
        if inspection:
            if not cls._complete_inspection(df, required_columns, start_date, end_date, cycle):
                raise ValueError("财务数据完整性检查未通过")

        return df

    @classmethod
    @validate_literal_params
    def get_rolling_financial(
            cls,
            code: str,
            cycle: FINANCIAL_CYCLE,
            start_date: str | None = None,
            end_date: str | None = None,
            required_columns: list[str] | None = None,
            inspection: bool = False
    ) -> pd.DataFrame:
        """
        获取标准化滚动财务数据 -1 bs取均值 -2 ps/cf 取累加值
        :param code: 企业代码
        :param start_date: 起始日期 (YYYY-MM-DD)
        :param end_date: 结束日期 (YYYY-MM-DD)
        :param cycle: 分析周期
        :param required_columns: 必须包含的财务科目
        :param inspection: 完备性价差
        :return: 处理后的财务数据DataFrame
        """
        # 参数处理
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # 读取原始数据
        bs = cls._read_parquet((DataPATH.FINANCIAL_DATA / f"{code}_bs").with_suffix(".parquet"))
        ps = cls._read_parquet((DataPATH.FINANCIAL_DATA / f"{code}_ps").with_suffix(".parquet"))
        cf = cls._read_parquet((DataPATH.FINANCIAL_DATA / f"{code}_cf").with_suffix(".parquet"))

        # 数据预处理
        # -1 周期过滤
        bs = cls._cycle_filter(bs, cycle)
        ps = cls._cycle_filter(ps, cycle)
        cf = cls._cycle_filter(cf, cycle)

        # -2 数据转换
        ps = cls._transform_to_non_cumulative(ps)
        cf = cls._transform_to_non_cumulative(cf)

        # -3 滚动取值
        window = ANNUALIZED_DAYS.get(cycle)

        bs_numeric_cols = bs.select_dtypes(include=['number']).columns
        bs[bs_numeric_cols] = bs[bs_numeric_cols].rolling(window=window, min_periods=window).mean()

        ps_numeric_cols = ps.select_dtypes(include=['number']).columns
        ps[ps_numeric_cols] = ps[ps_numeric_cols].rolling(window=window, min_periods=window).sum()

        cf_numeric_cols = cf.select_dtypes(include=['number']).columns
        cf[cf_numeric_cols] = cf[cf_numeric_cols].rolling(window=window, min_periods=window).sum()

        # -4 合并数据
        df = pd.concat([bs, ps, cf], axis=1).sort_index()

        # -5 过滤重复列名
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        # -6 时间过滤
        df = df[df.index >= start_date] if start_date is not None else df
        df = df[df.index <= end_date] if end_date is not None else df

        # -7 关键词过滤
        df = df[required_columns] if required_columns else df

        # -8 数据完整性检查
        if inspection:
            if not cls._complete_inspection(df, required_columns, start_date, end_date, cycle):
                raise ValueError("财务数据完整性检查未通过")

        return df

    @classmethod
    def get_bonus(
            cls,
            code: str
    ) -> pd.DataFrame:
        """
        获取分红数据
        :param code：代码
        :return: 分红数据
        """
        path = (DataPATH.BONUS_FINANCING_DATA / f"{code}_lnfhrz").with_suffix(".parquet")

        return cls._read_parquet(path)[["dividend"]]

    @classmethod
    def get_total_shares(
            cls,
            code: str,
            financial_date: pd.DatetimeIndex | None = None
    ) -> pd.DataFrame:
        """
        获取总股本
        :param code：代码
        :param financial_date：财务数据日期
        :return 总股本数据
        """
        # 读取文件
        path = (DataPATH.SHARES_DATA / code).with_suffix(".parquet")
        df = cls._read_parquet(path)

        if financial_date is None:
            return df

        # 填充缺失日期
        df = df.reindex(df.index.union(financial_date))
        # 排序并填充数据
        df = df.sort_index().ffill().bfill()
        # 截取 financial_date 对应的数据
        df = df.loc[financial_date]

        return df

    @classmethod
    @validate_literal_params
    def get_kline(
            cls,
            code: str,
            cycle: KLINE_CYCLE,
            adjusted_mode: KLINE_SHEET,
            start_date: str | None = None,
            end_date: str | None = None,
            aligned_to_month_end: bool = False,
    ) -> pd.DataFrame:
        """
        获取标准化行情数据
        :param code: 代码
        :param cycle: 周期
        :param adjusted_mode: 复权模式
        :param start_date: 起始日期
        :param end_date: 结束日期
        :param aligned_to_month_end: 为对齐财务日期，日期调整至月末（仅适用于月度及其以上数据）
        :return: 处理后的行情数据
        """
        # 读取文件
        path = (DataPATH.STOCK_KLINE_DATA / cycle / adjusted_mode / code).with_suffix(".parquet")
        df = cls._read_parquet(path)

        # 时间过滤
        df = df[df.index >= pd.to_datetime(start_date)] if start_date is not None else df
        df = df[df.index <= pd.to_datetime(end_date)] if end_date is not None else df

        # 日期调整至月末
        if aligned_to_month_end and cycle not in ALIGNED_TO_MONTH_END:
            df.index = cls._aligned_to_month_end(df.index)

        return df.sort_index()

    @classmethod
    @validate_literal_params
    def get_index_kline(
            cls,
            code: str,
            cycle: KLINE_CYCLE,
            start_date: str | None = None,
            end_date: str | None = None,
            aligned_to_month_end: bool = False
    ) -> pd.DataFrame:
        """
        获取标准化行情数据
        :param code: 代码
        :param cycle: 周期
        :param start_date: 起始日期
        :param end_date: 结束日期
        :param aligned_to_month_end: 为对齐财务日期，日期调整至月末（仅适用于月度及其以上数据）
        :return: 处理后的行情数据
        """
        # 读取数据
        path = (DataPATH.INDEX_KLINE_DATA / cycle / code).with_suffix(".parquet")
        df = cls._read_parquet(path)

        # 时间过滤
        df = df[df.index >= pd.to_datetime(start_date)] if start_date is not None else df
        df = df[df.index <= pd.to_datetime(end_date)] if end_date is not None else df

        # 日期调整至月末
        if aligned_to_month_end and cycle not in ALIGNED_TO_MONTH_END:
            df.index = cls._aligned_to_month_end(df.index)

        return df.sort_index()

    @classmethod
    @validate_literal_params
    def get_industry_codes(
            cls,
            sheet_name: INDUSTRY_SHEET,
            industry_info: dict[str, str],
            return_type: INDUSTRY_RETURN_TYPE = "dataframe"
    ) -> dict[str, list[str]] | list | pd.DataFrame:
        """
        获取行业分类对应的股票代码
        :param sheet_name: 工作表名 1）"A" 当前上市A股； 2）"Total_A" 包含已退市的上市A股； 3）"AHU" A/H/USA三地上市的国内企业
        :param industry_info: 行业信息字典 {行业名称: 行业级别}
        :param return_type: 返回数据类型 "dict", "list", "dataframe"
        :return: 行业-股票代码映射字典
        """
        # 路径检查
        path = DataPATH.INDUSTRY_CLASSIFICATION.with_suffix(".xlsx")
        path = cls._validate_path(path)

        # 读取文件
        df = pd.read_excel(path, sheet_name=sheet_name)
        # 数据过滤
        if "全部" in industry_info.keys():
            filter_df = df
        else:
            # 生成条件列表
            cond_list = [df[col] == value for value, col in industry_info.items()]
            combined_cond = reduce(or_, cond_list)
            filter_df = df[combined_cond]

        # 返回数据类型
        # -1 dataframe
        if return_type == "dataframe":
            return filter_df
        # -2 dict
        elif return_type == "dict":
            result = dict()
            for industry, level in industry_info.items():
                if industry == "全部":
                    grouped = df.groupby(level)
                    result.update(
                        {
                            str(k): [code.split(".")[0] for code in g["股票代码"].tolist()]
                            for k, g in grouped
                        }
                    )
                else:
                    mask = filter_df[level].str.contains(industry)
                    result[industry] = [code.split(".")[0] for code in filter_df[mask]["股票代码"].tolist()]
            return result
        # -3 list
        elif return_type == "list":
            return [code.split(".")[0] for code in filter_df["股票代码"].tolist()]
        else:
            raise ValueError(f"Invalid return_type: {return_type}")

    @classmethod
    @validate_literal_params
    def get_trading_calendar(
            cls,
            sheet_name: CYCLE,
            aligned_to_month_end: bool = False
    ) -> pd.DatetimeIndex:
        """
        获取交易日历
        :param sheet_name: 工作表名
        :param aligned_to_month_end: 为对齐财务日期，日期调整至月末（仅适用于月度及其以上数据）
        :return: 交易日历
        """
        path = DataPATH.TRADING_CALENDAR.with_suffix(".xlsx")

        date_index = cls._read_sheet(path, sheet_name).index

        # 日期调整至月末
        if aligned_to_month_end and sheet_name not in ALIGNED_TO_MONTH_END:
            date_index = cls._aligned_to_month_end(date_index)

        return date_index

    @classmethod
    def get_listed_nums(
            cls
    ) -> pd.DataFrame:
        """获取交易日历"""
        path = DataPATH.LISTED_NUMS.with_suffix(".xlsx")

        return cls._read_sheet(path)

    @classmethod
    def load_yaml_file(
            cls,
            file_path: Path,
            key: str
    ) -> dict[str, str]:
        """
        加载 YAML 文件并提取指定键的值
        :param file_path: YAML 文件路径
        :param key: 需要提取的键
        :return: 提取的字典数据
        """
        with open(file_path.with_suffix(".yaml"), encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get(key, {})

    @classmethod
    def load_parquet(
            cls,
            file_path: Path,
            set_date_index: bool = False
    ) -> pd.DataFrame:
        """
        读取 parquet 文件
        :param file_path: parquet 文件路径
        :param set_date_index: 设置日期索引
        :return: 提取的数据
        """
        path = file_path.with_suffix(".parquet")
        if set_date_index:
            return cls._read_parquet(path)

        path = cls._validate_path(path)
        return pd.read_parquet(path)
