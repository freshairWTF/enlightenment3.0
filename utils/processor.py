import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import Callable
from sklearn.preprocessing import PowerTransformer

from constant.type_ import ERROR, GROUP_MODE, validate_literal_params


###############################################################
class DataProcessor:
    """数据处理"""

    def __init__(self):
        self.winsorizer = Winsorizer                    # 缩尾
        self.dimensionless = Dimensionless              # 无量纲
        self.neutralization = Neutralization            # 中性化
        self.refactor = Refactor                        # 重构
        self.classification = Classification            # 分类


###############################################################
class Winsorizer:
    """缩尾（去极值）"""

    @staticmethod
    def mad(
            factor_values: pd.DataFrame | pd.Series,
            n: int = 3
    ) -> pd.DataFrame | pd.Series:
        """
        中位数绝对偏差值法：适用于非正态分布，异常值敏感性低
        将数据剪切到 [median - n*MAD, median + n*MAD] 范围内
        :param factor_values: 需要剪切的数据
        :param n：偏差倍数
        :return 去极值后的数据
        """
        def _process_series(s: pd.Series) -> pd.Series:
            median = s.median()
            # MAD 的尺度校正到与标准差一致
            mad = (s - median).abs().median() * 1.4826
            lower = median - n * mad
            upper = median + n * mad
            return s.clip(lower, upper)

        if isinstance(factor_values, pd.DataFrame):
            return factor_values.apply(_process_series)
        elif isinstance(factor_values, pd.Series):
            return _process_series(factor_values)
        else:
            raise TypeError('仅支持 pandas DataFrame/Series 类型输入')

    @staticmethod
    def sigma(
            factor_values: pd.DataFrame | pd.Series,
            n: int | float = 3,
            ddof: int = 1
    ) -> pd.DataFrame | pd.Series:
        """
        正态分布法（3σ原则）异常值处理：适用于正态分布，异常值敏感度高
        :param factor_values: 需要剪切的数据
        :param n：偏差倍数
        :param ddof: 标准差自由度调整，默认为1（样本标准差）1）0: 总体标准差；2）1: 样本标准差（默认）
        :return 去极值后的数据
        """
        def _process_series(s: pd.Series) -> pd.Series:
            mean, std = s.mean(), s.std(ddof=ddof)
            # 处理全零标准差情况
            if std == 0:
                return s.clip(lower=mean, upper=mean)
            lower = mean - n * std
            upper = mean + n * std
            return s.clip(lower, upper)

        if isinstance(factor_values, pd.DataFrame):
            return factor_values.apply(_process_series)
        elif isinstance(factor_values, pd.Series):
            return _process_series(factor_values)
        else:
            raise TypeError('仅支持 pandas DataFrame/Series 类型输入')

    @staticmethod
    def percentile(
            factor_values: pd.DataFrame | pd.Series,
            upper_q: float = 0.975,
            lower_q: float = 0.025
    ) -> pd.DataFrame | pd.Series:
        """
        百分位法异常值处理：适用于任意分布，异常值敏感度中
        将数据剪切到 [lower_q分位数, upper_q分位数] 范围内
        :param factor_values: 需要剪切的数据
        :param upper_q：上分位
        :param lower_q：下分位
        :return 去极值后的数据
        """
        def _process_series(s: pd.Series) -> pd.Series:
            lower_bound, upper_bound = s.quantile([lower_q, upper_q])
            # 处理无效分位数（如全相同值）
            if pd.isna(lower_bound) or pd.isna(upper_bound):
                return s
            # 确保分位数顺序正确
            lower, upper = sorted([lower_bound, upper_bound])
            return s.clip(lower, upper)

        if isinstance(factor_values, pd.DataFrame):
            return factor_values.apply(_process_series)
        elif isinstance(factor_values, pd.Series):
            return _process_series(factor_values)
        else:
            raise TypeError('仅支持 pandas DataFrame/Series 类型输入')


###############################################################
class Dimensionless:
    """无量纲"""

    @staticmethod
    def standardization(
            factor_values: pd.DataFrame | pd.Series,
            ddof: int = 1,
            error: ERROR = "raise"
    ) -> pd.DataFrame | pd.Series:
        """
        数据标准化（Z-Score 标准化）
        :param factor_values: 需要标准化的数据
        :param ddof: 标准差自由度调整，默认为1（样本标准差）1）0: 总体标准差；2）1: 样本标准差（默认）
        :param error: 异常应对策略
        :return 标准化后的数据
        """
        def _process_series(s: pd.Series) -> pd.Series:
            mean = s.mean()
            std = s.std(ddof=ddof)
            # 处理常量列
            if std == 0:
                if error == "raise":
                    raise ValueError(f'常量列无法标准化: {s.name}')
                else:
                    return pd.Series(np.nan, index=s.index, name=s.name)
            return (s - mean) / std

        if isinstance(factor_values, pd.DataFrame):
            return factor_values.apply(_process_series)
        elif isinstance(factor_values, pd.Series):
            return _process_series(factor_values)
        else:
            raise TypeError('仅支持 pandas DataFrame/Series 类型输入')

    @staticmethod
    def normalization(
            factor_values: pd.DataFrame | pd.Series,
            feature_range: tuple[float, float] = (0, 1),
            window: int | None = None,
            min_periods: int | None = None
    ) -> pd.DataFrame | pd.Series:
        """
        改进版归一化方法，支持滚动窗口归一化
        :param factor_values: 输入数据（DataFrame或Series）
        :param feature_range: 目标值域范围，默认(0,1)
        :param window: 滚动窗口大小（None表示全局归一化）
        :param min_periods: 窗口最小计算长度
        """
        def _process_series(s: pd.Series) -> pd.Series:
            # 滚动归一化
            if window:
                # 计算滚动极值
                rolling_min = s.rolling(window=window, min_periods=min_periods).min()
                rolling_max = s.rolling(window=window, min_periods=min_periods).max()
                # 处理分母（避免除零）
                denominator = (rolling_max - rolling_min).replace(0, np.nan)

                scale = (feature_range[1] - feature_range[0]) / denominator
                return (s - rolling_min) * scale + feature_range[0]
            # 全局归一化
            else:
                min_val, max_val = s.min(), s.max()
                if min_val == max_val:
                    return pd.Series(index=s.index, dtype='float64')  # 常量列返回空

                scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
                return (s - min_val) * scale + feature_range[0]

        # 应用处理逻辑
        if isinstance(factor_values, pd.DataFrame):
            return factor_values.apply(_process_series)
        elif isinstance(factor_values, pd.Series):
            return _process_series(factor_values)
        else:
            raise TypeError("仅支持 pandas DataFrame/Series 类型输入")


##############################################################
class NeutralizationDev:
    """
    中性化
        -1 底层中性化方法 哑变量/非哑变量 热编码？
        -2 扩展方法 因子中性化
        -3 接口 定制化中性化方法
    """

    # -----------------------------------
    # 底层中性化方法
    # -----------------------------------
    @staticmethod
    def __multi_factors(
            factor_values: pd.DataFrame | pd.Series,
            neutral_factors: pd.DataFrame,
            winsorizer: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series],
            dimensionless: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series],
    ):
        """多因子中性化"""
        def __sector_neutral(
                factor_series: pd.Series
        ) -> pd.Series:
            """截面中性"""
            # --------------------------
            # 数据预处理
            # --------------------------
            # 合并数据并丢弃缺失值
            combined = pd.concat([factor_series, neutral_factors], axis=1)
            combined.columns = ['factor', 'neutral_factor']
            combined = combined.dropna()
            if combined.empty:
                raise ValueError("有效数据量为零，无法进行计算")

            # --------------------------
            # 回归建模
            # --------------------------
            x = sm.add_constant(combined["neutral_factor"], has_constant='add')
            model = sm.OLS(combined['factor'], x).fit()

            # --------------------------
            # 残差处理
            # --------------------------
            residuals = pd.Series(model.resid, index=combined.index)
            neutralized = factor_series.copy()
            neutralized.loc[residuals.index] = residuals

            return neutralized

        # --------------------------
        # 输入校验
        # --------------------------
        if factor_values is None or neutral_factors is None:
            raise ValueError("输入数据不能为 None")

        if not isinstance(neutral_factors, pd.DataFrame | pd.Series):
            raise TypeError("industry_series 必须为 pd.DataFrame | pd.Series")

        if len(factor_values) != len(neutral_factors):
            raise ValueError("因子数据与行业数据长度不一致")

        # --------------------------
        # 市值数据预处理
        # --------------------------
        # 去极值 -> 标准化
        neutral_factors = (
            neutral_factors.
            pipe(winsorizer).
            pipe(dimensionless)
        )

        # --------------------------
        # 核心处理逻辑
        # --------------------------
        if isinstance(factor_values, pd.DataFrame):
            return factor_values.apply(lambda col: __sector_neutral(col))
        elif isinstance(factor_values, pd.Series):
            return __sector_neutral(factor_values)
        else:
            raise TypeError('仅支持 DataFrame/Series 类型输入')

    # @staticmethod
    # def __dump


##############################################################
class Neutralization:
    """中性化"""

    @staticmethod
    def linear_regression(
            factor_values: pd.DataFrame | pd.Series,
            neutral_factor: pd.Series,
            winsorizer: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series],
            dimensionless: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series],
    ) -> pd.DataFrame | pd.Series:
        """
        市值中性化
        :param factor_values: 待处理的因子数据，支持 DataFrame(多因子) 或 Series(单因子)
        :param neutral_factor: 回归因子
        :param winsorizer: 缩尾方法，优先使用 percentile
        :param dimensionless: 无量纲方法，优先使用 standardization
        :return 中性化后的数据
        """
        def __market_value_neutral(
                factor_series: pd.Series
        ) -> pd.Series:
            """处理单个因子序列"""
            # --------------------------
            # 数据预处理
            # --------------------------
            # 合并数据并丢弃缺失值
            combined = pd.concat([factor_series, neutral_factor], axis=1)
            combined.columns = ['factor', 'neutral_factor']
            combined = combined.dropna()
            if combined.empty:
                raise ValueError("有效数据量为零，无法进行计算")

            # --------------------------
            # 回归建模
            # --------------------------
            x = sm.add_constant(combined["neutral_factor"], has_constant='add')
            model = sm.OLS(combined['factor'], x).fit()

            # --------------------------
            # 残差处理
            # --------------------------
            residuals = pd.Series(model.resid, index=combined.index)
            neutralized = factor_series.copy()
            neutralized.loc[residuals.index] = residuals

            return neutralized

        # --------------------------
        # 输入校验
        # --------------------------
        if factor_values is None or neutral_factor is None:
            raise ValueError("输入数据不能为 None")

        if not isinstance(neutral_factor, pd.Series):
            raise TypeError("industry_series 必须为 pd.Series")

        if len(factor_values) != len(neutral_factor):
            raise ValueError("因子数据与行业数据长度不一致")

        # --------------------------
        # 市值数据预处理
        # --------------------------
        # 1. 去极值
        neutral_factor = winsorizer(neutral_factor)
        # 2. 标准化
        neutral_factor = dimensionless(neutral_factor)

        # --------------------------
        # 核心处理逻辑
        # --------------------------
        if isinstance(factor_values, pd.DataFrame):
            return factor_values.apply(lambda col: __market_value_neutral(col))
        elif isinstance(factor_values, pd.Series):
            return __market_value_neutral(factor_values)
        else:
            raise TypeError('仅支持 DataFrame/Series 类型输入')

    @staticmethod
    def log_market_cap(
            factor_values: pd.DataFrame | pd.Series,
            market_value: pd.Series,
            winsorizer: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series],
            dimensionless: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series],
    ) -> pd.DataFrame | pd.Series:
        """
        市值中性化
        :param factor_values: 待处理的因子数据，支持 DataFrame(多因子) 或 Series(单因子)
        :param market_value: 对数市值
        :param winsorizer: 缩尾方法，优先使用 percentile
        :param dimensionless: 无量纲方法，优先使用 standardization
        :return 中性化后的数据
        """
        def __market_value_neutral(
                factor_series: pd.Series
        ) -> pd.Series:
            """处理单个因子序列"""
            # --------------------------
            # 数据预处理
            # --------------------------
            # 合并数据并丢弃缺失值
            combined = pd.concat([factor_series, market_value_processed], axis=1)
            combined.columns = ['factor', 'market_value']
            combined = combined.dropna()
            if combined.empty:
                raise ValueError("有效数据量为零，无法进行计算")

            # --------------------------
            # 回归建模
            # --------------------------
            x = sm.add_constant(combined["market_value"], has_constant='add')
            model = sm.OLS(combined['factor'], x).fit()

            # --------------------------
            # 残差处理
            # --------------------------
            residuals = pd.Series(model.resid, index=combined.index)
            neutralized = factor_series.copy()
            neutralized.loc[residuals.index] = residuals

            return neutralized

        # --------------------------
        # 输入校验
        # --------------------------
        if factor_values is None or market_value is None:
            raise ValueError("输入数据不能为 None")

        if not isinstance(market_value, pd.Series):
            raise TypeError("industry_series 必须为 pd.Series")

        if len(factor_values) != len(market_value):
            raise ValueError("因子数据与行业数据长度不一致")

        # --------------------------
        # 市值数据预处理
        # --------------------------
        # 1. 去极值
        market_value_processed = winsorizer(market_value)
        # 2. 标准化
        market_value_processed = dimensionless(market_value_processed)

        # --------------------------
        # 核心处理逻辑
        # --------------------------
        if isinstance(factor_values, pd.DataFrame):
            return factor_values.apply(lambda col: __market_value_neutral(col))
        elif isinstance(factor_values, pd.Series):
            return __market_value_neutral(factor_values)
        else:
            raise TypeError('仅支持 DataFrame/Series 类型输入')

    @classmethod
    def industry(
            cls,
            factor_values: pd.DataFrame | pd.Series,
            industry_series: pd.Series
    ) -> pd.DataFrame | pd.Series:
        """
        行业中性化处理
        通过线性回归去除因子中的行业影响，返回残差作为中性化后的因子值
        :param factor_values: 待处理的因子数据，支持 DataFrame(多因子) 或 Series(单因子)
        :param industry_series: 行业分类数据，必须为 pandas Series
        :return: 行业中性化后的因子数据，与输入类型一致
        """
        def __industry_neutral(
                factor_series: pd.Series,
                error: ERROR = "raise"
        ) -> pd.Series:
            """
            行业中性化处理
            :param factor_series: 因子值
            :param error: 异常应对策略
            """
            # --------------------------
            # 数据预处理
            # --------------------------
            # 合并数据并丢弃缺失值
            combined = pd.concat([factor_series, industry_series], axis=1)
            combined.columns = ['factor', 'industry']
            combined = combined.dropna()

            if combined.empty:
                raise ValueError("有效数据量为零，无法进行计算")

            # --------------------------
            # 生成哑变量 drop_first=True 避免共线性
            # --------------------------
            dummy_matrix = pd.get_dummies(
                combined['industry'],
                prefix='ind',
                drop_first=True,
                dtype=float
            )
            # 移除零方差列（增强鲁棒性）
            dummy_matrix = dummy_matrix.loc[:, dummy_matrix.std() > 0]

            # --------------------------
            # 回归建模
            # --------------------------
            try:
                # 添加常数项
                x = sm.add_constant(dummy_matrix, has_constant='add')
                model = sm.OLS(combined['factor'], x).fit()
            except Exception as e:
                return cls.__handle_error(f"回归失败: {e}", error, factor_series)

            # --------------------------
            # 残差处理
            # --------------------------
            residuals = pd.Series(model.resid, index=combined.index)
            neutralized = factor_series.copy()
            neutralized.loc[residuals.index] = residuals

            return neutralized

        # --------------------------
        # 输入校验
        # --------------------------
        if factor_values is None or industry_series is None:
            raise ValueError("输入数据不能为 None")

        if not isinstance(industry_series, pd.Series):
            raise TypeError("industry_series 必须为 pandas Series")

        if len(factor_values) != len(industry_series):
            raise ValueError("因子数据与行业数据长度不一致")

        # --------------------------
        # 核心处理逻辑
        # --------------------------
        if isinstance(factor_values, pd.DataFrame):
            return factor_values.apply(lambda col: __industry_neutral(col, industry_series))
        elif isinstance(factor_values, pd.Series):
            return __industry_neutral(factor_values, industry_series)
        else:
            raise TypeError('仅支持 DataFrame/Series 类型输入')

    @classmethod
    def __handle_error(
            cls,
            msg: str,
            error: str,
            original: pd.Series
    ) -> pd.Series:
        """统一处理错误信息"""
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(msg)
        return original.copy()


###############################################################
class Refactor:
    """重构"""

    @staticmethod
    def restructure_factor(
            factor_value: pd.Series,
            denominator_value: pd.Series,
            min_positive: float = 1e-6
    ) -> pd.Series:
        """
        因子重构: -1 正值化 -2 对数化 -3 因子与构成因子的分母截面回归，取残差
        :param factor_value: 因子序列
        :param denominator_value: 构成因子的分母序列
        :param min_positive: 正值化偏移量
        :return: 重构因子
        """
        # --------------------------
        # 数据预处理
        # --------------------------
        # 合并数据并丢弃缺失值
        combined = pd.concat([factor_value, denominator_value], axis=1)
        combined.columns = ['factor', 'denominator']
        combined = combined.dropna()
        if combined.empty:
            raise ValueError("有效数据量为零，无法进行计算")

        # 正值化处理（避免log(0)）
        combined["factor"] = combined["factor"] + min_positive + np.abs(combined["factor"].min())
        combined["denominator"] = combined["denominator"] + min_positive + np.abs(combined["denominator"].min())

        # 对数转换
        combined["factor"] = np.log(combined["factor"])
        combined["denominator"] = np.log(combined["denominator"])

        # --------------------------
        # 回归建模
        # --------------------------
        x = sm.add_constant(combined["denominator"])
        y = combined["factor"]
        model = sm.OLS(y, x, missing='drop').fit()

        # --------------------------
        # 残差处理
        # --------------------------
        # 保留原始索引
        residuals = pd.Series(model.resid, index=combined.index)

        # 重新对齐原始索引 (处理可能存在缺失值的情况)
        return factor_value.copy().loc[residuals.index] * 0 + residuals

    @staticmethod
    def symmetric_orthogonal(
            factor_values: pd.DataFrame
    ) -> pd.DataFrame:
        """
        因子对称正交，消除因子间相关性
        :param factor_values: 因子数据
        :return 正交化后的因子
        """
        # 输入数据校验
        if factor_values.empty:
            return factor_values.copy()

        # 转换为numpy数组并中心化
        x = factor_values.values.astype(np.float64)
        x -= x.mean(axis=0)  # 中心化处理

        # 计算协方差矩阵
        try:
            cov = x.T @ x / (x.shape[0] - 1)  # 无偏估计
        except Exception as e:
            raise ValueError(f"协方差矩阵计算失败: {str(e)}") from e

        # 特征分解（使用更稳定的eigh）
        try:
            eigen_values, eigen_vectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"特征分解失败: {str(e)}") from e

        # 处理特征值
        idx = np.argsort(eigen_values)[::-1]  # 降序排列索引
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

        # 处理负特征值（数值稳定性）
        eigen_values = np.maximum(eigen_values, 0.0)

        # 正则化处理（避免除以零）
        epsilon = 1e-8
        sqrt_inv = 1.0 / np.sqrt(eigen_values + epsilon)

        # 构造正交变换矩阵
        try:
            transform = eigen_vectors @ np.diag(sqrt_inv) @ eigen_vectors.T
        except Exception as e:
            raise ValueError(f"变换矩阵构造失败: {str(e)}") from e

        # 应用变换
        try:
            x_ortho = x @ transform
        except Exception as e:
            raise ValueError(f"矩阵乘法失败: {str(e)}") from e

        # 重建DataFrame
        return pd.DataFrame(
            x_ortho,
            index=factor_values.index,
            columns=factor_values.columns
        )

    @staticmethod
    def shift_factors_value(
            factor_values: pd.DataFrame,
            fixed_col: list[str],
            lag_periods: int = 1,
    ) -> pd.DataFrame:
        """
        将每个DataFrame中的因子向后移指定期数，用于 T-N期因子 与 T期涨跌幅 的拟合回归
        :param factor_values: 原始数据
        :param fixed_col: 固定因子
        :param lag_periods: 滞后期数
        :return: 平移后的数据
        """
        # 深拷贝原始数据
        copied_data = factor_values.copy(deep=True).reset_index(drop=True)
        # 确定需要平移的列
        shifted_col = copied_data.columns.difference(fixed_col).tolist()

        grouped = copied_data.sort_values('date').groupby('股票代码')
        shifted_factors = grouped[shifted_col].transform(
            lambda x: x.shift(lag_periods)
        )

        # 将平移后的数据替换原始因子列
        copied_data[shifted_col] = shifted_factors

        # 删除因子列中存在NaN的行（至少有一个因子为NaN即删除）
        result_data = copied_data.dropna(
            subset=shifted_col,
            how="any"
        ).reset_index(drop=True)

        return result_data

    @classmethod
    def shift_factors_value_for_dict(
            cls,
            raw_data: dict[str, pd.DataFrame],
            lag_periods: int,
            factors_col: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        将每个DataFrame中的因子向后移指定期数，用于 T-N期因子 与 T期涨跌幅 的拟合回归
        :param raw_data: 原始数据
        :param lag_periods: 滞后期数
        :param factors_col: 因子名
        :return: 平移后的数据
        """
        # 深拷贝原始数据
        copied_data = {k: v.copy(deep=True) for k, v in raw_data.items()}

        # 确定需要平移的列
        if factors_col is None:
            sample_df = next(iter(copied_data.values()))
            factors_col = sample_df.columns.difference(["pctChg"]).tolist()
        else:
            factors_col = [f for f in factors_col if f != "pctChg"]

        # 无有效列直接返回
        if not factors_col:
            return copied_data

        # 合并所有数据并添加临时日期标记
        combined = pd.concat(
            {date: df[factors_col] for date, df in copied_data.items()},
            names=["date"]
        ).reset_index(level="date")

        # 按资产分组，滞后平移
        shifted = combined.groupby(combined.index)[factors_col].shift(lag_periods)
        combined[factors_col] = shifted

        # 按日期拆分回字典
        result = {}
        sorted_dates = sorted(copied_data.keys(), key=lambda x: pd.to_datetime(x))
        for date in sorted_dates:
            df = combined[combined.date == date].drop(columns="date")
            df = df.reindex(copied_data[date].index)

            result_date = copied_data[date].copy()
            result_date[factors_col] = df[factors_col]
            result_date = result_date.dropna(subset=factors_col, how="any")

            if not result_date.empty:
                result[date] = result_date

        return result

    @classmethod
    def box_cox_transfer(
            cls,
            factor_value: pd.DataFrame,
            standardize: bool = False
    ) -> pd.Series:
        """
        box-cox变换（适用于严格为正的数据，0数据可以加上一个极小正值）
        :param factor_value: 因子数值
        :param standardize: 是否标准化处理
        :return: box-cox变换后的因子数值
        """
        # -1 数据平移
        factor_value = factor_value - np.min(factor_value) + 1e-5

        # -2 估计λ
        tf = PowerTransformer(method="box-cox", standardize=standardize)
        tf.fit(factor_value)

        # -3 转换原始数据
        result = tf.transform(factor_value)

        return result

    @classmethod
    def yeo_johnson_transfer(
            cls,
            factor_value: pd.DataFrame,
            standardize: bool = False
    ) -> pd.Series:
        """
        Yeo-Johnson变换（允许负数）
        :param factor_value: 因子数值
        :param standardize: 是否标准化处理
        :return: Yeo-Johnson变换后的因子数值
        """
        # -1 估计λ
        tf = PowerTransformer(method="yeo-johnson", standardize=standardize)
        tf.fit(factor_value)

        # -2 转换原始数据
        result = tf.transform(factor_value)

        return result


###############################################################
class Classification:

    @classmethod
    @validate_literal_params
    def divide_into_group(
            cls,
            factor_values: pd.DataFrame,
            factor_col: str,
            processed_factor_col: str,
            group_mode: GROUP_MODE,
            group_nums: int,
            group_label: list[str],
            negative: bool = False
    ) -> pd.DataFrame:
        """
        分组 -1等距 distant；-2 等频 frequency
        :param factor_values: 分组数据
        :param factor_col: 分组因子列名
        :param processed_factor_col: 分组预处理因子列名
        :param group_mode: 分组模式
        :param group_nums: 分组数
        :param group_label: 分组标签
        :param negative: 负值单列
        """
        method = {
            "distant": cls.__distant,
            "frequency": cls.__frequency
        }.get(group_mode, cls.__frequency)

        result_df = []
        for date, df in factor_values.groupby("date"):
            try:
                if df.shape[0] >= group_nums:
                    df["group"] = method(
                        df, factor_col, processed_factor_col, group_nums, group_label, negative
                    )
                    result_df.append(df)
            except ValueError:
                continue

        return pd.concat(result_df)

    @staticmethod
    def __distant(
            df: pd.DataFrame,
            factor_col: str,
            processed_factor_col: str,
            group_nums: int,
            group_label: list[str],
            negative: bool
    ) -> pd.Series:
        """等距分组"""
        if negative:
            negative_mask = df[factor_col] < 0
            df["group"] = np.where(
                negative_mask,
                'negative',
                pd.NA
            )
            non_negative_group = pd.cut(
                df.loc[~negative_mask, processed_factor_col],
                bins=group_nums,
                labels=group_label,
                duplicates="drop"
            )
            df.loc[~negative_mask, "group"] = non_negative_group
            return df["group"].astype("category")
        else:
            return pd.cut(
                pd.to_numeric(df[processed_factor_col]),
                bins=group_nums,
                labels=group_label,
                duplicates="drop"
            )

    @staticmethod
    def __frequency(
            df: pd.DataFrame,
            factor_col: str,
            processed_factor_col: str,
            group_nums: int,
            group_label: list[str],
            negative: bool
    ) -> pd.Series:
        """等频分组"""
        if negative:
            negative_mask = df[factor_col] < 0
            df["group"] = np.where(
                negative_mask,
                'negative',
                pd.NA
            )
            non_negative_group = pd.qcut(
                df.loc[~negative_mask, processed_factor_col],
                q=group_nums,
                labels=group_label,
                duplicates="drop"
            )
            df.loc[~negative_mask, "group"] = non_negative_group
            return df["group"].astype("category")
        else:
            return pd.qcut(
                pd.to_numeric(df[processed_factor_col]),
                q=group_nums,
                labels=group_label,
                duplicates="drop"
            )
