import pandas as pd

from base_model import MultiFactorsModel
from constant.type_ import GROUP_MODE, FACTOR_WEIGHT, POSITION_WEIGHT, validate_literal_params
from utils.quant_processor import QuantProcessor


########################################################################
class LinearMultiFactors(MultiFactorsModel):
    """线性多因子模型"""

    model_name: str = "线性多因子模型"

    @validate_literal_params
    def __init__(
            self,
            raw_data: dict[str, pd.DataFrame],
            factors_name: dict[str, list[str]],
            group_nums: int,
            group_label: list[str],
            group_mode: GROUP_MODE = "frequency",
            factor_weight_method: FACTOR_WEIGHT = "equal",
            factor_weight_window: int = 12,
            position_weight_method: POSITION_WEIGHT = "equal",
            position_distribution: tuple[float, float] = (1, 1),
            individual_position_limit: float = 0.1,
            index_data: dict[str, pd.DataFrame] | None = None,
    ):
        """
        :param raw_data: 数据
        :param factors_name: 因子名
        :param group_nums: 分组数
        :param group_mode: 分组模式
        :param factor_weight_method: 因子权重方法
        :param factor_weight_window: 因子权重窗口数
        :param position_weight_method: 仓位权重方法
        :param position_distribution: 仓位集中度
        :param individual_position_limit: 单一持仓上限
        :param index_data: 指数数据
        """
        self.raw_data = raw_data
        self.factors_name = factors_name
        self.group_nums = group_nums
        self.group_label = group_label
        self.group_mode = group_mode
        self.index_data = index_data
        self.factor_weight_method = factor_weight_method
        self.factor_weight_window = factor_weight_window
        self.position_weight_method = position_weight_method
        self.position_distribution = position_distribution

        self.individual_position_limit = individual_position_limit,

    def run(self):
        """
        线性模型：
            1）计算因子权重；
            2）选择加权方法，计算综合Z-Score
            3）Z-Score回归/计算预期收益率；
            4）分组；
            5）仓位权重；
        """
        # -1 因子权重
        factor_weights = self.factor_weight.get_weights(
            factors_data=self.raw_data,
            factors_name=self.factors_name,
            method=self.factor_weight_method,
            window=self.factor_weight_window
        )

        # -2 综合Z值
        z_score = self.calc_z_scores(
            data=self.raw_data,
            factors_name=self.factors_name,
            weights=factor_weights
        )

        # -3 预期收益率
        predict_return = self.calc_predict_return(
            x_value=z_score,
            y_value={date: df["pctChg"] for date, df in self.raw_data.items()},
            window=self.factor_weight_window
        )

        # -4 分组
        grouped_data = QuantProcessor.divide_into_group(
            predict_return,
            factor_col="",
            processed_factor_col="predicted",
            group_mode=self.group_mode,
            group_nums=self.group_nums,
            group_label=self.group_label,
        )

        # -5 仓位权重
        position_weight = self.position_weight.get_weights(
            grouped_data,
            factor_name="predicted",
            method=self.position_weight_method,
            distribution=self.position_distribution
        )

        # -6 数据合并
        result = self.join_data(grouped_data, position_weight)
        result = self.join_data(result, z_score)
        result = self.join_data(result, self.raw_data)

        return result
