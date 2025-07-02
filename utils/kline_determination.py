"""K线形态判定"""
import numpy as np
import pandas as pd

from decimal import Decimal, ROUND_HALF_UP


###############################################################
class KlineDetermination:
    """k线形态判定"""

    @staticmethod
    def decimal_round(x):
        return Decimal(str(x)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)

    @staticmethod
    def set_board(
            code: str
    ) -> str:
        """
        根据代码设置标签
        :param code: 企业代码
        -1 688 科创板 -2 300/301/303 创业板 -3 其他 主板
        """
        conditions = [
            code[:3] == "688",
            code[:3] in ["300", "301", "303"],
            code[:2] == "60"
        ]
        choices = [
            "科创板",
            "创业板",
            "沪市主板"
        ]
        return str(np.select(conditions, choices, default="深市主板"))

    @classmethod
    def positive_line(
            cls,
            df: pd.DataFrame,
            min_change: float = 0.0,
            upper_shadow_bounds: tuple[float, float] = None,
            lower_shadow_bounds: tuple[float, float] = None
    ) -> pd.Series:
        """
        阳线判定（带涨幅阈值、上下影线）
        :param df: 必须包含open, close, pctChg列的DataFrame
        :param min_change: 最小涨幅阈值（正数百分比，如0.02表示2%跌幅）
        :param upper_shadow_bounds: 上影线区间控制
                                    例：(0.1, 0.5)表示上影线在实体10%-50%之间
                                    None表示不限制
        :param lower_shadow_bounds: 下影线区间控制
        """
        # 参数检验
        required_columns = {'open', 'close', 'high', 'low', 'pctChg', 'preclose'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"缺少必要字段: {missing}")

        # k线判定
        is_positive = pd.Series(df['close'] > df['preclose'], index=df.index)
        entity = df['close'] - df['open']

        # 上影线控制
        if upper_shadow_bounds is not None:
            # 上影线长度
            upper_shadow = df['high'] - df['close']
            # 上下比例
            lower, upper = upper_shadow_bounds
            if lower is not None: lower = entity * lower
            if upper is not None: upper = entity * upper
            # 区间验证
            if lower is not None: is_positive &= (upper_shadow >= lower)
            if upper is not None: is_positive &= (upper_shadow <= upper)

        # 下影线控制
        if lower_shadow_bounds is not None:
            # 下影线长度
            lower_shadow = df['open'] - df['low']
            # 上下比例
            lower, upper = lower_shadow_bounds
            if lower is not None: lower = entity * lower
            if upper is not None: upper = entity * upper
            # 区间验证
            if lower is not None: is_positive &= (lower_shadow >= lower)
            if upper is not None: is_positive &= (lower_shadow <= upper)

        # 涨幅控制
        if min_change:
            is_positive &= (df["pctChg"] > min_change)

        return is_positive

    @classmethod
    def negative_line(
            cls,
            df: pd.DataFrame,
            min_change: float = 0.0,
            upper_shadow_bounds: tuple[float, float] = None,
            lower_shadow_bounds: tuple[float, float] = None
    ) -> pd.Series:
        """阴线判定（带跌幅阈值）
        :param df: 必须包含open, close, pctChg列的DataFrame
        :param min_change: 最小跌幅阈值（正数百分比，如0.02表示2%跌幅）
        :param upper_shadow_bounds: 上影线区间控制
                            例：(0.1, 0.5)表示上影线在实体10%-50%之间
                            None表示不限制
        :param lower_shadow_bounds: 下影线区间控制
        """
        # 参数检验
        required_columns = {'open', 'close', 'high', 'low', 'pctChg', 'preclose'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"缺少必要字段: {missing}")

        # k线判定
        is_negative = pd.Series(df['close'] < df['preclose'], index=df.index)
        entity = df['open'] - df['close']

        # 上影线控制
        if upper_shadow_bounds is not None:
            # 上影线长度
            upper_shadow = df['high'] - df['open']
            # 上下比例
            lower, upper = upper_shadow_bounds
            if lower is not None: lower = entity * lower
            if upper is not None: upper = entity * upper
            # 区间验证
            if lower is not None: is_negative &= (upper_shadow >= lower)
            if upper is not None: is_negative &= (upper_shadow <= upper)

        # 下影线控制
        if lower_shadow_bounds is not None:
            # 下影线长度
            lower_shadow = df['close'] - df['low']
            # 上下比例
            lower, upper = lower_shadow_bounds
            if lower is not None: lower = entity * lower
            if upper is not None: upper = entity * upper
            # 区间验证
            if lower is not None: is_negative &= (lower_shadow >= lower)
            if upper is not None: is_negative &= (lower_shadow <= upper)

        # 跌幅控制
        if min_change > 0:
            is_negative &= (df['pctChg'] < -min_change)

        return is_negative

    @classmethod
    def explosive_quantity(
            cls,
            df: pd.DataFrame,
            window: int = 1,
            min_change: float = 1.0
    ) -> pd.Series:
        """
        成交量爆量
        :param df: 必须包含volume列的DataFrame
        :param window: 比对均值窗口数
        :param min_change: 最小变动幅度
        """
        # 参数检验
        if 'volume' not in df.columns:
            raise ValueError("输入数据必须包含volume列")
        if window < 1:
            raise ValueError("窗口长度至少为1日")
        if min_change < 1.0:
            raise ValueError("min_change应大于等于1")

        # 计算历史均值（排除当日）
        historical_ma = (
            df['volume']
            .rolling(window=window, min_periods=2)
            .mean()
            .shift(1)
        )

        return (df['volume'] > historical_ma * min_change).fillna(False)

    @classmethod
    def reduced_quantity(
            cls,
            df: pd.DataFrame,
            window: int = 1,
            min_change: float = 0.999
    ) -> pd.Series:
        """
        成交量缩量
        :param df: 必须包含volume列的DataFrame
        :param window: 比对均值窗口数
        :param min_change: 最小变动幅度
        """
        # 参数检验
        if 'volume' not in df.columns:
            raise ValueError("输入数据必须包含volume列")
        if window < 1:
            raise ValueError("窗口长度至少为1日")
        if min_change > 1.0:
            raise ValueError("min_change应小于1")

        # 计算历史均值（排除当日）
        historical_ma = (
            df['volume']
            .rolling(window=window, min_periods=2)
            .mean()
            .shift(1)
        )

        return (df['volume'] < historical_ma * min_change).fillna(False)

    # 涨停、跌停
    @classmethod
    def limit_up(
            cls,
            df: pd.DataFrame,
            board: str
    ) -> pd.Series:
        """
        涨停
        :param df: k线数据
        :param board: 板块
        """
        df = df[["date", "isST", "preclose", "close"]].copy()
        df = df.set_index("date")

        # -------------------------
        # 涨跌幅度
        # -------------------------
        # -1 科创板 20%
        if board == "科创板":
            df["limit_rate"] = 0.2
        # -2 创业板
        # 注册制 10% -> 20% | ST 5% -> 20%
        elif board == "创业板":
            cond = [
                (df.index >= "2020-08-24"),
                (df.index < "2020-08-24") & (df["isST"] == 0),
                (df.index < "2020-08-24") & (df["isST"] == 1),
            ]
            choices = [
                0.2,
                0.1,
                0.05
            ]
            df["limit_rate"] = np.select(cond, choices)
        # -3 主板
        # 非ST 10% | ST 5%
        else:
            df["limit_rate"] = np.where(df["isST"] == 0, 0.1, 0.05)

        # -------------------------
        # 涨停板判定
        # -------------------------
        pre_close = df["preclose"]

        limit_price = (pre_close * (1 + df["limit_rate"]))
        # 将浮点数格式化为足够小数位的字符串，消除中间误差
        limit_price = limit_price.map(lambda x: f"{x:.6f}").astype(str)
        # 采用Decimal，消除银行家舍入规则的偏差
        limit_price = limit_price.apply(cls.decimal_round).astype(df["close"].dtype)

        return (round(df["close"], 2) == limit_price).astype(int)

    @classmethod
    def limit_down(
            cls,
            df: pd.DataFrame,
            board: str
    ) -> pd.Series:
        """
        跌停
        :param df: k线数据
        :param board: 板块
        """
        df = df[["date", "isST", "preclose", "close"]].copy()
        df = df.set_index("date")

        # -------------------------
        # 涨跌幅度
        # -------------------------
        # -1 科创板 20%
        if board == "科创板":
            df["limit_rate"] = 0.2
        # -2 创业板
        # 注册制 10% -> 20% | ST 5% -> 20%
        elif board == "创业板":
            cond = [
                (df.index >= "2020-08-24"),
                (df.index < "2020-08-24") & (df["isST"] == 0),
                (df.index < "2020-08-24") & (df["isST"] == 1),
            ]
            choices = [
                0.2,
                0.1,
                0.05
            ]
            df["limit_rate"] = np.select(cond, choices)
        # -3 主板
        # 非ST 10% | ST 5%
        else:
            df["limit_rate"] = np.where(df["isST"] == 0, 0.1, 0.05)

        # -------------------------
        # 跌停板判定
        # -------------------------
        pre_close = df["preclose"]

        limit_price = (pre_close * (1 - df["limit_rate"]))
        # 将浮点数格式化为足够小数位的字符串，消除中间误差
        limit_price = limit_price.map(lambda x: f"{x:.6f}").astype(str)
        # 采用Decimal，消除银行家舍入规则的偏差
        limit_price = limit_price.apply(cls.decimal_round).astype(df["close"].dtype)

        return (round(df["close"], 2) == limit_price).astype(int)

    @classmethod
    def n_consecutive_mask(
            cls,
            series: pd.Series,
            n: int = 1
    ) -> pd.Series:
        """标记连续N个True的情况"""
        return (
            series
            .rolling(window=n, min_periods=n)
            .apply(lambda x: x.all())
            .fillna(0)
            .astype(bool)
        )
