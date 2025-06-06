"""范围过滤"""
import pandas as pd

from constant.type_ import FILTER_MODE, CYCLE, validate_literal_params


####################################################
class RangeFilter:
    """
    名单过滤：
        -1 基础过滤
        -2 白名单
        -3 全部股票
        -4 整体股份
        -5 大盘股
        -6 超级大盘
        -7 超级小盘
    """

    PARAMETERS = ['所有者权益', 'isST', 'volume', 'close', '市值']

    @validate_literal_params
    def __init__(
            self,
            data: dict[str: pd.DataFrame],
            filter_mode: FILTER_MODE,
            cycle: CYCLE
    ):
        """
        :param data: 待过滤数据
        :param filter_mode: 过滤模式
        :param cycle: 周期
        """
        self.data = data
        self.filter_mode = filter_mode
        self.window = self._setup_window(cycle)

        # 预处理股票首次出现索引 用于次新股筛选
        self.stock_first_index = self._preprocess_stock_first_index()

    # --------------------------
    # 其他 方法
    # --------------------------
    def _validate_parameters(
            self,
            df: pd.DataFrame
    ) -> bool:
        """参数验证装饰器"""
        missing = [p for p in self.PARAMETERS if p not in df.columns]
        if missing:
            print(f"Missing parameters: {missing}")
            return False
        return True

    @classmethod
    def _apply_filters(
            cls,
            df: pd.DataFrame,
            filters: list
    ) -> pd.DataFrame:
        """应用多个过滤条件"""
        for filter_func in filters:
            df = filter_func(df)
        return df

    @classmethod
    def _setup_window(
            cls,
            cycle: str
    ) -> int:
        return {
            'day': 250,
            'week': 52,
            'month': 12,
            'quarter': 4,
            'half': 2,
            'year': 1
        }.get(cycle, 0)

    def _preprocess_stock_first_index(
            self
    ) -> dict[str, int]:
        """预处理每个股票的首次出现索引"""
        stock_first_index = {}
        for idx, (date, df) in enumerate(self.data.items()):
            for stock in df.index:
                if stock not in stock_first_index:
                    stock_first_index[stock] = idx

        return stock_first_index

    # --------------------------
    # 过滤模式
    # --------------------------
    def _white_filter(
            self
    ) -> list[callable]:
        """白名单过滤"""
        return [
            self.__net_asset,
            self.__st,
            self.__suspension,
            self.__par_value
        ]

    def _entire_filter(
            self
    ) -> list[callable]:
        """
        全部股票：过滤T日停牌
        """
        return [
            self.__suspension
        ]

    def _overall_filter(
            self
    ) -> list[callable]:
        """
        整体股票：1）过滤市值最小的15%的微型股份；
                2）过滤T日停牌和已退市的股票；
                3）过滤ST；
                4）过滤次新股；
                5）过滤净资产为负；
                6）过滤低于面值
        """
        return [
            lambda df: self.__market_value_quantile(df, 0.15, True),
            self.__net_asset,
            self.__st,
            self.__suspension,
            self.__par_value
        ]

    def _mega_cap_filter(
            self
    ) -> list[callable]:
        """
        超级大盘：整体股票中，市值排名前6%的股票
        """
        return [
            lambda df: self.__market_value_quantile(df, 0.94, True),
            self.__net_asset,
            self.__st,
            self.__suspension,
            self.__par_value
        ]

    def _large_cap_filter(
            self
    ) -> list[callable]:
        """
        大盘：整体股票中，市值排名前16%的股票
        """
        return [
            lambda df: self.__market_value_quantile(df, 0.84, True),
            self.__net_asset,
            self.__st,
            self.__suspension,
            self.__par_value
        ]

    def _small_cap_filter(
            self
    ) -> list[callable]:
        """
        超级小盘：整体股票中，总市值排名后20%的股票
        """
        return [
            lambda df: self.__market_value_quantile(df, 0.2, False),
            self.__net_asset,
            self.__st,
            self.__suspension,
            self.__par_value
        ]

    # --------------------------
    # 过滤方法
    # --------------------------
    def __sub_new(
            self,
            df: pd.DataFrame,
            required_index: int
    ) -> pd.DataFrame:
        """次新股过滤：通过预处理的首发索引快速过滤"""
        valid_stocks = [
            stock for stock in df.index
            if self.stock_first_index.get(stock, float("inf")) <= required_index
        ]
        return df.loc[valid_stocks]

    @classmethod
    def __suspension(
            cls,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """停牌过滤：当月成交量为0"""
        return df[df['volume'] != 0]

    @classmethod
    def __st(
            cls,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """st过滤"""
        return df[df['isST'] == 0]

    @classmethod
    def __net_asset(
            cls,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """净资产过滤"""
        return df[df['所有者权益'] > 0]

    @classmethod
    def __par_value(
            cls,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """面值过滤：收盘价 > 1元"""
        return df[df['close'] >= 1]

    @classmethod
    def __market_value_quantile(
            cls,
            df: pd.DataFrame,
            percentage: float,
            gt=True
    ) -> pd.DataFrame:
        """
        市值排名过滤
        :param percentage: 百分比
        :param gt: 方向
        """
        return df[df['市值'] > df['市值'].quantile(percentage)] if gt else df[df['市值'] < df['市值'].quantile(percentage)]

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(
            self
    ) -> dict[str, pd.DataFrame]:
        """执行过滤"""
        result = {}
        for i, (date, df) in enumerate(self.data.items()):
            if not self._validate_parameters(df):
                continue

            filters = getattr(self, self.filter_mode)()

            # 处理次新股过滤条件
            if self.filter_mode != "_entire_filter" and i >= self.window:
                required_index = i - self.window
                filters.append(lambda x, ri=required_index: self.__sub_new(x, ri))

            filtered_df = self._apply_filters(df, filters)
            if not filtered_df.empty:
                result[date] = filtered_df

        return result
