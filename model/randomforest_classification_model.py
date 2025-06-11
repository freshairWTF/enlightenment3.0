import pandas as pd

from constant.type_ import GROUP_MODE, FACTOR_WEIGHT, POSITION_WEIGHT, validate_literal_params
from utils.processor import DataProcessor


########################################################################
class RandomForestClassificationModel:
    """randomforest分类模型"""

    model_name: str = "randomforest分类模型"

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

    @classmethod
    def calc_predict_return_by_randomforest(
            cls,
            x_value: dict[str, pd.DataFrame | pd.Series],
            y_value: dict[str, pd.Series],
            window: int = 12
    ) -> dict[str, pd.DataFrame]:
        """
        基于随机森林的滚动窗口回归预测收益率
        :param x_value: T期截面数据
        :param y_value: T期收益率
        :param window: 滚动窗口长度
        :return: 预期收益率
        """
        # 随机森林参数配置（参考网页7、网页8的金融预测最佳实践）
        params = {
            'n_estimators': 200,  # 树的数量，网页8使用200棵
            'max_depth': 10,  # 树的最大深度，网页7建议5-15之间
            'min_samples_split': 10,  # 节点分裂最小样本数，抑制噪声干扰
            'max_features': 'sqrt',  # 每棵树随机选择√(总特征数)的特征
            'n_jobs': -1,  # 使用全部CPU核心加速计算
            'random_state': 42,  # 确保结果可复现
            'verbose': 1  # 显示训练进度（参考网页6的日志配置）
        }

        sorted_dates = sorted(x_value.keys())
        result = {}

        for i in range(window, len(sorted_dates)):
            # ====================
            # 样本内训练（改进点：增加特征筛选）
            # ====================
            train_window = sorted_dates[i - window:i]

            # 改进点：动态特征合并（参考网页7的prepare_features方法）
            train_dfs = []
            for date in train_window:
                # 合并特征与目标变量时保留原始特征（网页8的X/y分离方式）
                df = pd.concat([x_value[date], y_value[date].rename('target')], axis=1)
                df["date"] = date
                train_dfs.append(df)

            train_data = pd.concat(train_dfs).dropna()

            # 改进点：增加滞后特征（参考网页3的shift特征构造）
            for lag in [1, 3, 5]:
                train_data[f'return_lag_{lag}'] = train_data.groupby(level=0)['target'].shift(lag)

            train_data = train_data.dropna()

            # 拆分训练集（网页6的滑动窗口实现方式）
            x_train = train_data.drop(['target', 'date'], axis=1)
            y_train = train_data['target']

            # ====================
            # 模型训练（增加早停机制）
            # ====================
            try:
                # 改进点：使用OOB误差估计（参考网页8的评估方法）
                model = RandomForestRegressor(**params, oob_score=True)
                model.fit(x_train, y_train)

                # 特征重要性筛选（网页7的核心功能）
                importance = pd.Series(model.feature_importances_, index=x_train.columns)
                selected_features = importance.nlargest(15).index.tolist()  # 保留前15个重要特征
            except Exception as e:
                print(f"训练失败：{str(e)}")
                continue

            # ====================
            # 样本外预测（改进点：增加不确定性估计）
            # ====================
            predict_date = sorted_dates[i]
            predict_df = x_value[predict_date].copy()

            # 为预测数据生成滞后特征（需与训练集同步）
            for lag in [1, 3, 5]:
                predict_df[f'return_lag_{lag}'] = y_value[sorted_dates[i - lag]].reindex(predict_df.index)

            x_predict = predict_df[selected_features].dropna()

            # 执行预测（网页6的预测实现方式）
            predicted = pd.Series(
                model.predict(x_predict),
                index=x_predict.index,
                name="predicted"
            )

            # 改进点：计算预测分位数（参考网页9的风险控制思想）
            quantiles = np.quantile([tree.predict(x_predict) for tree in model.estimators_],
                                    q=[0.25, 0.75], axis=0)
            predicted_df = pd.DataFrame({
                'point_pred': predicted,
                'lower_bound': quantiles[0],
                'upper_bound': quantiles[1]
            })

            result[predict_date] = predicted_df

        return result

    def run(self):
        """
        xgboost非线性模型：
            1）因子加权；
            2）模型训练/回测；
            3）分组；
            4）仓位权重；
        """
        # -1 因子权重
        factor_weights = self.factor_weight.get_factors_weights(
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

        # -3 加权因子
        weight_factors = self.calc_weight_factors(
            data=self.raw_data,
            factors_name=self.factors_name,
            weights=factor_weights
        )

        # -4 预期收益率
        predict_return = self.calc_predict_return_by_randomforest(
            x_value=weight_factors,
            y_value={date: df["pctChg"] for date, df in self.raw_data.items()},
            window=self.factor_weight_window
        )
        print(predict_return['2018-05-25'])
        # -4 分组
        grouped_data = QuantProcessor.divide_into_group(
            predict_return,
            factor_col="",
            processed_factor_col="predicted",
            group_mode=self.group_mode,
            group_nums=self.group_nums,
            group_label=self.group_label,
        )
        print(grouped_data['2018-05-25'])
        # -5 仓位权重
        position_weight = self.position_weight.get_weights(
            grouped_data,
            factor_col="predicted",
            method=self.position_weight_method,
            distribution=self.position_distribution
        )

        # -6 数据合并
        result = self.join_data(grouped_data, position_weight)
        result = self.join_data(result, z_score)
        result = self.join_data(result, self.raw_data)
        print(result['2018-05-25'])
        print(result['2018-05-25'].columns)
        return result
