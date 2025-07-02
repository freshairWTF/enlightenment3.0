"""标的池过滤"""
import pandas as pd

from constant.type_ import FILTER_MODE, CYCLE, validate_literal_params


####################################################
class StockPoolFilter:
    """
    股票池过滤：
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
            filter_mode: FILTER_MODE,
            cycle: CYCLE
    ):
        """
        :param filter_mode: 过滤模式
        :param cycle: 周期
        """
        self.filter_mode = filter_mode
        self.window = self._setup_window(cycle)

    def __call__(
            self,
            target_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        数据管道可调用接口
        :param target_data: 未过滤数据
        :return: 过滤数据
        """
        return self.run(target_data)

    @classmethod
    def _setup_window(
            cls,
            cycle: str
    ) -> pd.DateOffset:
        """次新股间隔时间"""
        offset_map = {
            'day': pd.DateOffset(days=250),
            'week': pd.DateOffset(weeks=52),
            'month': pd.DateOffset(months=12),
            'quarter': pd.DateOffset(months=3),
            'half': pd.DateOffset(months=6),
            'year': pd.DateOffset(years=1)
        }
        return offset_map.get(cycle, pd.DateOffset(days=0))

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
    # 过滤方法
    # --------------------------
    def _apply_filters(
            self,
            group: pd.DataFrame,
            listing_date: dict[str, pd.Timestamp]
    ) -> pd.DataFrame:
        """
        应用过滤条件到每个日期分组
        :param group: 分组数据
        :param listing_date: 上市时间
        """
        filters = getattr(self, self.filter_mode)()

        # 次新股过滤
        if self.filter_mode != "entire":
            current_date = group['date'].iloc[0]
            min_date = (current_date - self.window).to_pydatetime().date()

            valid_stocks = [
                stock for stock in group['股票代码']
                if listing_date.get(stock, pd.Timestamp.max) <= min_date
            ]
            group = group[group['股票代码'].isin(valid_stocks)]

        for filter_func in filters:
            group = filter_func(group)

        return group

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def run(
            self,
            target_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        执行过滤
        :param target_data: 未过滤数据
        :return: 过滤数据
        """
        # 股票上市时间 用于次新股筛选
        listing_date = target_data.groupby('股票代码')['date'].min().to_dict()

        # 按日期分组处理
        filtered_groups = []
        for _, group in target_data.groupby('date'):
            filtered = self._apply_filters(group, listing_date)
            if not filtered.empty:
                filtered_groups.append(filtered)

        return pd.concat(filtered_groups).reset_index(drop=True)
