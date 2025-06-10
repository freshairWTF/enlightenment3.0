import pandas as pd

from constant.type_ import GROUP_MODE, FACTOR_WEIGHT, POSITION_WEIGHT, validate_literal_params
from utils.processor import DataProcessor


########################################################################
class XGBoostRegressionModel:
    """xgboost回归模型"""

    model_name: str = "xgboost回归模型"

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
    def calc_predict_return_by_xgboost(
            cls,
            x_value: dict[str, pd.DataFrame | pd.Series],
            y_value: dict[str, pd.Series],
            window: int = 12
    ) -> dict[str, pd.DataFrame]:
        """
        滚动窗口回归预测收益率（集成Hyperopt自动调参）
        :param x_value: T期截面数据
        :param y_value: T期收益率
        :param window: 滚动窗口长度
        :return: 预期收益率
        """

        # 定义XGBoost参数搜索空间 [1,6](@ref)
        # def get_param_space():
        #     return {
        #         'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        #         'max_depth': hp.quniform('max_depth', 3, 12, 1),  # 加深深度
        #         'subsample': hp.uniform('subsample', 0.5, 0.95),
        #         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.95),
        #         'reg_alpha': hp.uniform('reg_alpha', 0, 1),  # L1上限降至0.3
        #         'reg_lambda': hp.uniform('reg_lambda', 0, 1),  # L2下限提升
        #         'gamma': hp.uniform('gamma', 0, 0.5)  # 分裂阈值压缩
        #     }

        def get_param_space():
            base_space = {
                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
                'subsample': hp.uniform('subsample', 0.7, 0.9)
            }
            if prev_model is None:  # 首期全量调参
                full_space = {
                    'max_depth': hp.quniform('max_depth', 3, 12, 1),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.95),
                    'reg_alpha': hp.uniform('reg_alpha', 0, 0.3),  # L1上限压缩
                    'reg_lambda': hp.uniform('reg_lambda', 0.7, 1),  # L2下限提升
                    'gamma': hp.uniform('gamma', 0, 0.3)
                }
                return {**base_space, **full_space}
            return base_space  # 后续仅优化关键参数

        # 目标函数（含时间序列验证）[2,7](@ref)
        def objective(params, X_train, y_train):

            tscv = TimeSeriesSplit(n_splits=3)
            losses = []

            # 时间序列交叉验证
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                try:
                    model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        ** {k: int(v) if k in ['max_depth'] else v for k, v in params.items()}
                    ).fit(X_tr, y_tr)
                    pred = model.predict(X_val)
                    losses.append(mean_squared_error(y_val, pred))
                except:
                    return {'loss': np.inf, 'status': STATUS_OK}

            return {'loss': np.mean(losses), 'status': STATUS_OK}

        def ic_score(y_true, y_pred):
            return np.corrcoef(y_true, y_pred)[0, 1]

        # 主流程
        sorted_dates = sorted(x_value.keys())
        result = {}
        trials_cache = None  # 用于增量调参的缓存
        prev_model = None

        for i in range(window, len(sorted_dates)):
            # ====================
            # 样本内数据准备
            # ====================
            train_window = sorted_dates[i - window:i]

            # 拼接训练数据（保持原有逻辑）
            train_dfs = []
            for date in train_window:
                df = pd.concat([x_value[date], y_value[date]], axis=1, join="inner")
                df["date"] = date
                train_dfs.append(df)
            train_data = pd.concat(train_dfs).dropna()

            # xgboost2.0之后的版本，不需手动添加截距项
            y_train = train_data["pctChg"]
            x_train = train_data.drop(["pctChg", "date"], axis=1)

            # ====================
            # Hyperopt参数优化 [3,8](@ref)
            # ====================
            trials = Trials()

            best_params = fmin(
                fn=lambda p: objective(p, x_train, y_train),
                space=get_param_space(),
                algo=tpe.suggest,
                max_evals=50,  # 每期评估次数
                trials=trials,
                rstate=np.random.default_rng(seed=42) ,
                verbose=False
            )

            # 参数类型转换
            best_params = {
                'objective': 'reg:quantileerror',
                'quantile_alpha': 0.5,
                'learning_rate': best_params['learning_rate'],
                'max_depth': int(best_params['max_depth']),
                'subsample': best_params['subsample'],
                'colsample_bytree': best_params['colsample_bytree'],
                'reg_alpha': best_params['reg_alpha'],
                'reg_lambda': best_params['reg_lambda'],
                'gamma': best_params['gamma'],
                'eval_metric': ic_score,
            }

            # ====================
            # 模型训练与预测
            # ====================
            model = xgb.XGBRegressor(**best_params)
            model.fit(x_train, y_train, xgb_model=prev_model)
            prev_model = model.get_booster()

            # 预测处理（保持原有逻辑）
            predict_date = sorted_dates[i]
            predict_df = x_value[predict_date]
            predicted = pd.Series(
                model.predict(predict_df),
                index=predict_df.index,
                name="predicted"
            )
            print(predict_date)
            result[predict_date] = predicted.to_frame()

        return result

    def run(self):
        """
        xgboost非线性模型（因子数量要一致）：
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
        predict_return = self.calc_predict_return_by_xgboost(
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
            factor_name="predicted",
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
