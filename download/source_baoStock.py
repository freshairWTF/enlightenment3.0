"""
第三方数据库（证券宝）
"""

import baostock as bs
import pandas as pd
from functools import lru_cache
from pathlib import Path

from constant.download import TimeDimension, SheetName
from constant.type_ import KLINE_SHEET, validate_literal_params
from utils.kline_determination import KlineDetermination


###########################################################################
class BaoStockDownLoader:
    """baostock数据下载类"""

    # --------------------------
    # 类属性：配置参数
    # --------------------------
    _KLINE_FIELDS_MAP = {
        'stock_d': 'date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST',
        'stock_other': 'date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg',
        'index': 'date,code,open,high,low,close,preclose,volume,amount,pctChg'
    }

    _PREPROCESSING_MAP = {
        'profit': ['roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM'],
        'operation': ['NRTurnRatio', 'NRTurnDays', 'INVTurnRatio', 'INVTurnDays', 'CATurnRatio', 'AssetTurnRatio'],
        'growth': ['YOYEquity', 'YOYAsset', 'YOYNI', 'YOYEPSBasic', 'YOYPNI']
    }

    def __init__(self):
        self._login()

    # --------------------------
    # 初始化与上下文管理
    # --------------------------
    @staticmethod
    def _login():
        """登录 BaoStock"""
        try:
            login_result = bs.login()
            if login_result.error_code != '0':
                raise ConnectionError(f'登录失败: {login_result.error_msg}')
        except Exception as e:
            raise RuntimeError(f'BaoStock 登录异常: {str(e)}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        bs.logout()

    # --------------------------
    # 通用查询方法
    # --------------------------
    @staticmethod
    def _query_data(
        query_func: callable,
        params: dict,
        fields: list[str]
    ) -> pd.DataFrame:
        """
        通用数据查询方法
        :param query_func: baostock 查询函数（如 bs.query_history_k_data_plus）
        :param params: 查询参数（如 code, start_date 等）
        :param fields: 返回字段列表（用于验证）
        """
        # 执行查询
        rs = query_func(**params)
        if rs.error_code != '0':
            raise ValueError(f'查询失败: {rs.error_msg}')

        # 提取数据并转换为 DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)

        # 验证字段一致性
        if set(df.columns) != set(fields):
            raise ValueError(f'字段不匹配！预期: {fields}, 实际: {df.columns.tolist()}')

        # 数值类型转换
        numeric_cols = [col for col in df.columns if col not in ('date', 'code')]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce', downcast='float')

        return df

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def get_kline_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str,
        adjust_flag: str,
        index: bool = False
    ) -> pd.DataFrame:
        """
        获取K线数据（支持股票和指数）
        :param code: 代码（有前缀）
        :param start_date: 起始日期
        :param end_date: 结束日期
        :param frequency: 数据频次
        :param adjust_flag: 复权类型
        :param index: 是否为指数
        :return: k线数据
        """
        # 选择字段模板
        if index:
            fields = self._KLINE_FIELDS_MAP['index'].split(',')
        else:
            fields_key = 'stock_d' if frequency == 'd' else 'stock_other'
            fields = self._KLINE_FIELDS_MAP[fields_key].split(',')
        # 执行查询
        return self._query_data(
            query_func=bs.query_history_k_data_plus,
            params={
                'code': code,
                'fields': ','.join(fields),
                'start_date': start_date,
                'end_date': end_date,
                'frequency': frequency,
                'adjustflag': adjust_flag
            },
            fields=fields
        )

    @lru_cache(maxsize=100)
    def get_constituent(
            self,
            index: str
    ) -> pd.DataFrame:
        """
        获取指数成分股（支持缓存）
        :param index: 指数代码
        :return: 指数成分股
        """
        query_map = {
            'ih': bs.query_sz50_stocks,
            'if': bs.query_hs300_stocks,
            'ic': bs.query_zz500_stocks
        }
        if index not in query_map:
            raise ValueError(f'不支持的指数类型: {index}，可选: {list(query_map.keys())}')

        return self._query_data(
            query_func=query_map[index],
            params={},
            fields=['code', 'code_name']
        )

    def get_quarter_data(
            self,
            code: str,
            year: int | str,
            quarter: int | str,
            data_type: str
    ) -> pd.DataFrame:
        """
        获取季度数据（盈利/运营/成长）
        :param code: 代码（有前缀）
        :param year: 年份
        :param quarter: 季度
        :param data_type: 数据类型：1）profit；2）operation；3）growth
        """
        query_map = {
            'profit': bs.query_profit_data,
            'operation': bs.query_operation_data,
            'growth': bs.query_growth_data
        }
        if data_type not in query_map:
            raise ValueError(f'不支持的数据类型: {data_type}，可选: {list(query_map.keys())}')

        df = self._query_data(
            query_func=query_map[data_type],
            params={'code': code, 'year': year, 'quarter': quarter},
            fields=query_map[data_type]().fields
        )
        self._preprocessing(df, data_type)
        return df

    @lru_cache(maxsize=1)
    def get_all_stock_codes(
            self
    ) -> pd.Series:
        """获取全量股票代码（带缓存）"""
        df = self._query_data(
            query_func=bs.query_stock_industry,
            params={},
            fields=bs.query_stock_industry().fields
        )
        return df['code'].drop_duplicates()

    # --------------------------
    # 数据处理方法
    # --------------------------
    @staticmethod
    def _preprocessing(
            df: pd.DataFrame,
            data_type: str
    ) -> None:
        """
        数据预处理：数据类型转换为浮点型
        :param df: 待处理数据
        :param data_type: 数据类型
        :return: 处理好的数据
        """
        columns = BaoStockDownLoader._PREPROCESSING_MAP.get(data_type, [])
        numeric_cols = [col for col in columns if col in df.columns]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce', downcast='float')


#####################################################
class BaoStockCleaner:
    """baostock数据清洗类"""

    kd = KlineDetermination

    # --------------------------
    # 数据处理方法
    # --------------------------
    @classmethod
    def _read_day_k(
        cls,
        dir_: Path,
        code: str,
        adjusted_mode: KLINE_SHEET | None
    ) -> pd.DataFrame:
        """
        读取日K数据
        :param dir_: 根目录路径
        :param code: 代码（无后缀）
        :param adjusted_mode: 复权模式
        """
        file_path = (
            (dir_  / adjusted_mode / code).with_suffix(".parquet") if adjusted_mode
            else (dir_  / code).with_suffix(".parquet"))
        try:
            return pd.read_parquet(file_path)
        except FileNotFoundError as e:
            raise ValueError(f"原始日K数据文件不存在: {e}") from e

    @classmethod
    def _data_fill(
        cls,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        数据填充：针对下载自baoStock的源头数据，如日K、周K、月K
        :param df: 待处理的数据
        :return: 填充好的数据
        """
        # 除涨跌幅外的数据
        fill_cols = ["volume", "amount", "turn"]
        df[fill_cols] = df[fill_cols].fillna(0).ffill()

        return df

    @classmethod
    @validate_literal_params
    def _process_adjustment(
        cls,
        dir_: Path,
        code: str,
        day_k: pd.DataFrame,
        adjusted_mode: KLINE_SHEET
    ) -> pd.DataFrame:
        """
        处理复权因子
        :param dir_: 根目录路径
        :param code: 代码（无后缀）
        :param day_k: 原始日K数据
        :param adjusted_mode: 复权模式
        :return: 复权数据
        """
        try:
            file_path = (dir_ / "original_day" / adjusted_mode / code).with_suffix(".parquet")
            adjusted_df = pd.read_parquet(file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"复权数据文件不存在: {e}") from e
        # 计算复权因子
        adjusted_df = cls._adjust_factor(adjust=adjusted_df, unadjusted=day_k)
        return adjusted_df

    @classmethod
    def _adjust_factor(
            cls,
            unadjusted: pd.DataFrame,
            adjust: pd.DataFrame
    ) -> pd.DataFrame:
        """
        复权因子 = 当前复权价格 / 原始价格
        :param unadjusted: 未复权数据
        :param adjust: 复权数据
        :return: 带有复权因子的复权数据
        """
        adjust['adjust_factor'] = (adjust['close'] / unadjusted['close']).ffill()

        return adjust

    @classmethod
    def _limit_up_and_down(
            cls,
            df: pd.DataFrame,
            code: str,
            adjusted_mode: str,
            dir_: Path
    ) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
        """
        判定当日是否涨跌停
        """
        # 前复权计算判定
        if adjusted_mode == "split_adjusted":
            board = cls.kd.set_board(code)
            return cls.kd.limit_up(df, board), cls.kd.limit_down(df, board)
        # 后复权直接读取前复权数据
        else:
            split_adjusted_day_k = cls._read_day_k(
                dir_ / "day",
                code,
                adjusted_mode='split_adjusted'
            )
            return split_adjusted_day_k[["date", "limit_up"]], split_adjusted_day_k[["date", "limit_down"]]

    @staticmethod
    def _generate_group_keys(
            df: pd.DataFrame,
            dimension: str
    ) -> pd.Series:
        """生成分组键序列"""
        date_series = df["date"]

        if dimension == TimeDimension.WEEK.value:
            # ISO周规则处理跨年周：每年的第一周必须满足以下条件之一：1）包含该年的第一个星期四；2）至少包含该年的4天
            return date_series.dt.isocalendar().year.astype(str) + "-" + \
                date_series.dt.isocalendar().week.astype(str).str.zfill(2)

        elif dimension == TimeDimension.MONTH.value:
            return date_series.dt.to_period("M").astype(str)

        elif dimension == TimeDimension.QUARTER.value:
            return date_series.dt.to_period("Q").astype(str)

        elif dimension == TimeDimension.HALF.value:
            half = date_series.dt.month.sub(1).floordiv(6).add(1)
            return date_series.dt.year.astype(str) + "-H" + half.astype(str)

        elif dimension == TimeDimension.YEAR.value:
            return date_series.dt.year.astype(str)

        else:
            raise ValueError(f"Unsupported time dimension: {dimension}")

    @classmethod
    @validate_literal_params
    def _synthesis_kline(
            cls,
            df: pd.DataFrame,
            dimension: str,
            adjusted_mode: KLINE_SHEET | None,
    ) -> pd.DataFrame:
        """
        将日K合并为月K、季K、半年K、年K
        :param df：日K
        :param dimension：合成维度
        :param adjusted_mode: 复权模式
        :return: (df) synthesis_df 合成K线
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 生成分组键
        group_keys = cls._generate_group_keys(df, dimension)
        group_df = df.groupby(group_keys, observed=True)
        synthesis_df = group_df.agg({
            "date": "last",
            "code": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "preclose": "first",
            "volume": "sum",
            "amount": "sum",
            "positive_line": "sum",
            "negative_line": "sum",
            **({"adjust_factor": "last"} if "adjust_factor" in df.columns else {})
        })

        # 计算涨跌幅
        # # 涨跌幅 = 最后一根收盘价/第一根开盘价 - 1
        synthesis_df["pctChg"] = synthesis_df["close"] / synthesis_df["open"] - 1
        # 实际涨跌幅 = 最后一根收盘价/第一根前收盘价 - 1
        synthesis_df["real_pctChg"] = synthesis_df["close"] / synthesis_df["preclose"] - 1

        # 处理非指数数据
        if adjusted_mode is not None:
            if "turn" in df.columns:
                synthesis_df["turn"] = group_df["turn"].sum()
            # 1：正常交易；2：停牌：只要有停牌就显示停牌
            if "tradestatus" in df.columns:
                synthesis_df["tradestatus"] = group_df["tradestatus"].min()
            # 1：ST；0：非ST：只要有ST就显示ST
            if "isST" in df.columns:
                synthesis_df["isST"] = group_df["isST"].max()
            # 复权因子标记
            if "adjustflag" in df.columns:
                synthesis_df["adjustflag"] = group_df["adjustflag"].first()
            # 复权因子，NaN向后填充
            if adjusted_mode in [SheetName.BACKWARD.value, SheetName.FORWARD.value]:
                synthesis_df["adjust_factor"] = group_df["adjust_factor"].last()
                synthesis_df["limit_up"] = group_df["limit_up"].sum()
                synthesis_df["limit_down"] = group_df["limit_down"].sum()

        # 重置索引
        synthesis_df.reset_index(drop=True, inplace=True)
        synthesis_df["date"] = synthesis_df["date"].dt.strftime("%Y-%m-%d")

        return synthesis_df

    # --------------------------
    # 公开 API 方法
    # --------------------------
    @classmethod
    @validate_literal_params
    def cleaning_stock(
        cls,
        dir_: Path,
        code: str,
        adjusted_mode: KLINE_SHEET
    ) -> dict:
        """
        数据清洗：以原始日K合成k线
            1）填充；2）增加st；3）合成
        :param dir_: 根目录路径
        :param code: 代码（无后缀）
        :param adjusted_mode: 复权模式
        """
        # ---------------------------------------
        # 读取模块
        # ---------------------------------------
        # 读取原始日k数据
        original_day_k = cls._read_day_k(
            dir_ / "original_day",
            code,
            adjusted_mode='unadjusted'
        )
        # fillna
        original_day_k = cls._data_fill(original_day_k)

        # ---------------------------------------
        # 计算模块
        # ---------------------------------------
        # 计算复权因子
        adjust_day_k = (
            cls._process_adjustment(dir_, code, original_day_k, adjusted_mode)
            if adjusted_mode in [SheetName.BACKWARD.value, SheetName.FORWARD.value]
            else original_day_k
        )

        # 判定涨跌停板
        if adjusted_mode in [SheetName.BACKWARD.value, SheetName.FORWARD.value]:
            limit_up, limit_down = cls._limit_up_and_down(
                adjust_day_k,
                code,
                adjusted_mode,
                dir_
            )
            if adjusted_mode == SheetName.FORWARD.value:
                adjust_day_k["limit_up"], adjust_day_k["limit_down"] = limit_up.values, limit_down.values
            elif adjusted_mode == SheetName.BACKWARD.value:
                adjust_day_k = pd.merge(adjust_day_k, limit_up.merge(limit_down, on='date'), on='date', how='inner')

        # 判断阳线/阴线
        adjust_day_k["positive_line"] = cls.kd.positive_line(adjust_day_k).astype("int")
        adjust_day_k["negative_line"] = cls.kd.negative_line(adjust_day_k).astype("int")

        # ---------------------------------------
        # 数据合成模块
        # ---------------------------------------
        kline_dict = {
            freq: cls._synthesis_kline(adjust_day_k, freq, adjusted_mode)
            for freq in ["week", "month", "quarter", "half", "year"]
        }
        kline_dict.update({'day': adjust_day_k})

        return kline_dict

    @classmethod
    def cleaning_index(
        cls,
        dir_: Path,
        code: str,
        adjusted_mode = None
    ) -> dict:
        """
        数据清洗：以原始日K合成k线
            1）填充；2）增加st；3）合成
        :param dir_: 根目录路径
        :param code: 代码（无后缀）
        :param adjusted_mode: 复权模式
        """
        # 读取原始日k数据
        original_day_k = cls._read_day_k(
            dir_ / "original_day",
            code,
            adjusted_mode
        )

        # 判断阳线/阴线
        original_day_k["positive_line"] = cls.kd.positive_line(original_day_k).astype("int")
        original_day_k["negative_line"] = cls.kd.negative_line(original_day_k).astype("int")

        # 数据合成
        kline_dict = {
            freq: cls._synthesis_kline(original_day_k, freq, adjusted_mode)
            for freq in ["week", "month", "quarter", "half", "year"]
        }
        kline_dict.update({'day': original_day_k})

        return kline_dict
