"""xgboost分类模型"""
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from template import ModelTemplate


########################################################################
class XGBoostClassificationModel(ModelTemplate):
    """
    XGBoost分类模型
        booster	'gbtree'	基学习器类型	使用树模型（CART）作为基础分类器，若需线性模型可改为 'gblinear'
        max_depth	6	树的最大深度	值越大模型越复杂（易过拟合），典型值范围 3–10
        n_estimators	100	弱学习器数量	树的数量越多模型越强（但可能过拟合），需与学习率平衡
        learning_rate	0.3	学习率/步长	值越小需更多树迭代，典型值 0.01–0.2
        gamma	0	分裂最小损失下降阈值	值越大分裂越保守（正则化作用），常用值 0–1
        min_child_weight	1	叶子节点样本权重和下限	值越大分裂越保守（防过拟合），典型值 1–10
        subsample	1	样本采样比例	<1 时可降低过拟合风险（如 0.8）
        colsample_bytree	1	特征采样比例	<1 时增强多样性（如 0.7–0.9）
        reg_alpha	0	L1 正则化系数	>0 时增强稀疏性（适合高维特征）
        reg_lambda	1	L2 正则化系数	控制权重平滑度（防过拟合）
        objective	'multi:softmax'	多分类目标函数	需额外指定 num_class（类别数），否则报错
    """

    def __init__(
            self,
            input_df: pd.DataFrame,
            model_setting: dataclass,
            descriptive_factors: list[str],
            index_data: dict[str, pd.DataFrame] | None = None,
    ):
        """
        :param input_df: 数据
        :param model_setting: 模型设置
        :param descriptive_factors: 描述性统计因子
        :param index_data: 指数数据
        """
        super().__init__(
            input_df,
            model_setting,
            descriptive_factors
        )
        self.index_data = index_data

        # 分类器参数
        self.keep_cols.append("group")
        self.le = LabelEncoder()                                                # 标签转换实例

    def _pre_processing(
            self,
            input_df: pd.DataFrame,
            prefix: str = "processed"
    ) -> pd.DataFrame:
        """
        数据预处理
            -1 缩尾
            -2 标准化
            -3 中性化
            -4 缩尾
            -5 标准化
        :param input_df: 初始数据
        :param prefix: 预处理生成因子前缀
        :return: 处理过的数据
        """
        def __process_single_date(
                df_: pd.DataFrame,
        ) -> pd.DataFrame:
            """单日数据处理"""
            for setting in self.factors_setting:
                factor_name = setting.factor_name
                processed_col = f"{prefix}_{factor_name}"
                df_[processed_col] = df_[factor_name].copy()

                # -1 方向变化
                if setting.reverse:
                    df_[processed_col] *= -1

                # -2 正态变换
                if setting.transfer:
                    df_[processed_col] = self.processor.refactor.yeo_johnson_transfer(df_[processed_col])

                # -3 第一次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

                # -4 中性化
                if setting.market_value_neutral:
                    df_[processed_col] = self.processor.neutralization.log_market_cap(
                        df_[processed_col],
                        df_["对数市值"],
                        winsorizer=self.processor.winsorizer.percentile,
                        dimensionless=self.processor.dimensionless.standardization
                    )
                if setting.industry_neutral:
                    df_[processed_col] = self.processor.neutralization.industry(
                        df_[processed_col],
                        df_["行业"]
                    )

                # -5 第二次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

            return df_

        result_dfs = []
        for date, group_df in input_df.groupby("date"):
            try:
                processed_df = __process_single_date(group_df)
                result_dfs.append(processed_df)
            except ValueError:
                continue

        # 合并处理结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    def model_training_and_predict(
            self,
            input_df: pd.DataFrame,
            x_cols: list[str],
            y_col: str,
            window: int = 12
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        模型训练与预测
        :param input_df: 输入数据
        :param x_cols: 因子列名
        :param y_col: 目标列名
        :param window: 滚动窗口长度
        :return: 预期收益率
        """
        # 按日期排序并转换为列表
        sorted_dates = sorted(input_df["date"].unique())

        # 评估指标
        metrics = []

        # 滚动窗口遍历
        result_dfs = []
        for i in range(window, len(sorted_dates)):
            # ====================
            # 样本内训练
            # ====================
            # -1 获取训练窗口数据
            train_window = sorted_dates[i - window: i]
            # -2 获取训练数据
            x_train, y_train = (
                input_df.loc[input_df["date"].isin(train_window), x_cols],
                input_df.loc[input_df["date"].isin(train_window), y_col]
            )
            # -4 标签转换 离散字符 -> 连续整数
            y_train = pd.Series(
                self.le.fit_transform(y_train),
                index=y_train.index
            )

            # -5 训练模型
            model = XGBClassifier(objective='multi:softmax')
            model.fit(
                x_train,
                y_train,
            )

            # ====================
            # 样本外预测
            # ====================
            # -1 获取预测日数据
            predict_date = sorted_dates[i]
            # -2 获取测试窗口数据
            true_df = input_df.loc[input_df["date"] == predict_date]
            x_true, y_true = (
                input_df.loc[input_df["date"] == predict_date, x_cols],
                input_df.loc[input_df["date"] == predict_date, y_col]
            )
            # -3 模型预测
            y_predict = model.predict(x_true)
            # -4 标签转换 连续整数 -> 离散字符
            true_df["group"] = pd.Series(
                self.le.inverse_transform(y_predict),
                index=true_df.index
            )

            # ====================
            # 权重优化（仓位权重、实际股数）
            # ====================
            # -1 数据转换
            portfolio_df = input_df.loc[input_df["date"].isin(
                sorted_dates[i - window: i+1]),
                ["date", "股票代码", "close", "行业", "volume", "pctChg"]
            ]
            price_df = portfolio_df.pivot(
                index="date",
                columns="股票代码",
                values="pctChg"
            )
            volume_df = portfolio_df.pivot(
                index="date",
                columns="股票代码",
                values="volume"
            )
            industry_df = portfolio_df.set_index("股票代码")["行业"]
            # -2 权重（分组）
            weights_series = []
            alloc_series = []
            for _, group_df in true_df.groupby("group"):
                weights, alloc = self.portfolio_optimizer(
                    price_df=price_df[group_df["股票代码"].tolist()].ffill().bfill(),
                    volume_df=volume_df[group_df["股票代码"].tolist()],
                    industry_df=industry_df[industry_df.index.isin(group_df["股票代码"].tolist())],
                    allocation=True if predict_date == sorted_dates[-1] and self.model_setting.generate_trade_file else False,
                    last_price_series=portfolio_df.pivot(index="date", columns="股票代码", values="close").iloc[-1]
                )
                weights_series.append(weights)
                if predict_date == sorted_dates[-1] and not alloc.empty:
                    alloc_series.append(alloc)

            # -3 合并
            true_df = pd.merge(
                true_df,
                pd.concat(weights_series).rename('position_weight'),
                left_on='股票代码',
                right_index=True,
                how='left'
            )
            if predict_date == sorted_dates[-1] and alloc_series:
                true_df = pd.merge(
                    true_df,
                    pd.concat(alloc_series).rename('买入股数'),
                    left_on='股票代码',
                    right_index=True,
                    how='left'
                )

            # ====================
            # 归因分析（多分类）
            # ====================
            shap_df = self.shap_for_multiclass(
                model,
                factors_name=x_cols,
                x_true=x_true,
                y_true=self.le.transform(y_true),
                y_predict=y_predict,
            )
            true_df = pd.concat([true_df, shap_df], axis=1)

            # ====================
            # 数据整合（原值、预测收益率/分组、仓位权重、实际股数）
            # ====================
            result_dfs.append(true_df)

            # ====================
            # 模型评估
            # ====================
            try:
                metrics_series = self.calculate_classification_metrics(y_true, true_df["group"])
                metrics_series.name = predict_date
                metrics.append(metrics_series)
            except ValueError:
                continue

        return (pd.concat(result_dfs, ignore_index=True),
                pd.concat(metrics, axis=1).mean(axis=1).to_frame(name="value").T)

    def run(
            self
    ) -> dict[str, pd.DataFrame]:
        """
        线性模型处理流程：
            -1 因子数值处理
            -2 因子升维/降维
            -3 模型训练、预测、分组
        """
        # ----------------------------------
        # 数值处理
        # ----------------------------------
        # 数据预处理
        self.input_df = self._pre_processing(self.input_df)
        # 标签预设
        self.input_df = self.processor.classification.divide_into_group(
            self.input_df,
            processed_factor_col="pctChg",
            group_mode=self.model_setting.group_mode,
            group_nums=self.model_setting.group_nums,
            group_label=self.model_setting.group_label,
            group_col="origin_group"
        )

        # ----------------------------------
        # 因子降维
        # ----------------------------------
        level_2_df = self._factors_synthesis(
            self.input_df,
            mode="THREE_TO_TWO"
        )

        # ----------------------------------
        # 因子相关性
        # ----------------------------------
        corr_df = self.calculate_factors_corr(
            factors_df=level_2_df,
            factors_name=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist()
        )

        # ----------------------------------
        # 模型
        # ----------------------------------
        pred_df, estimate_metric = self.model_training_and_predict(
            input_df=level_2_df,
            x_cols=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist(),
            y_col="origin_group",
            window=self.model_setting.factor_weight_window
        )

        return {
            "模型": pred_df,
            "模型评估": estimate_metric,
            "因子相关性": corr_df,
            "因子shap值": pred_df.filter(regex=r'shap_|^date$|^group$', axis=1)
        }


########################################################################
class XGBoostClassificationCVModel(ModelTemplate):
    """XGBoost分类模型（超参调优）"""

    def __init__(
            self,
            input_df: pd.DataFrame,
            model_setting: dataclass,
            descriptive_factors: list[str],
            index_data: dict[str, pd.DataFrame] | None = None,
    ):
        """
        :param input_df: 数据
        :param model_setting: 模型设置
        :param descriptive_factors: 描述性统计因子
        :param index_data: 指数数据
        """
        super().__init__(
            input_df,
            model_setting,
            descriptive_factors
        )
        self.index_data = index_data

        # 分类器参数
        self.keep_cols.append("group")
        self.le = LabelEncoder()                                                # 标签转换实例

        # 超参数网格
        # n_estimators/learning_rate
        # max_depth/min_child_weight/gamma
        # reg_alpha/reg_lambda
        # subsample/colsample_bytree
        self.model_param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [10, 50, 100, 500, 1000, 3000, 5000]
        }

    def _pre_processing(
            self,
            input_df: pd.DataFrame,
            prefix: str = "processed"
    ) -> pd.DataFrame:
        """
        数据预处理
            -1 缩尾
            -2 标准化
            -3 中性化
            -4 缩尾
            -5 标准化
        :param input_df: 初始数据
        :param prefix: 预处理生成因子前缀
        :return: 处理过的数据
        """
        def __process_single_date(
                df_: pd.DataFrame,
        ) -> pd.DataFrame:
            """单日数据处理"""
            for setting in self.factors_setting:
                factor_name = setting.factor_name
                processed_col = f"{prefix}_{factor_name}"
                df_[processed_col] = df_[factor_name].copy()

                # -1 方向变化
                if setting.reverse:
                    df_[processed_col] *= -1

                # -2 正态变换
                if setting.transfer:
                    df_[processed_col] = self.processor.refactor.yeo_johnson_transfer(df_[processed_col])

                # -3 第一次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

                # -4 中性化
                if setting.market_value_neutral:
                    df_[processed_col] = self.processor.neutralization.log_market_cap(
                        df_[processed_col],
                        df_["对数市值"],
                        winsorizer=self.processor.winsorizer.percentile,
                        dimensionless=self.processor.dimensionless.standardization
                    )
                if setting.industry_neutral:
                    df_[processed_col] = self.processor.neutralization.industry(
                        df_[processed_col],
                        df_["行业"]
                    )

                # -5 第二次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

            return df_

        result_dfs = []
        for date, group_df in input_df.groupby("date"):
            try:
                processed_df = __process_single_date(group_df)
                result_dfs.append(processed_df)
            except ValueError:
                continue

        # 合并处理结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    def model_training_and_predict(
            self,
            input_df: pd.DataFrame,
            x_cols: list[str],
            y_col: str,
            window: int = 12
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        模型训练与预测
        :param input_df: 输入数据
        :param x_cols: 因子列名
        :param y_col: 目标列名
        :param window: 滚动窗口长度
        :return: 预期收益率
        """
        """
        测试下过采样
        以及 分类加权 哪种效果好
        """
        # 按日期排序并转换为列表
        sorted_dates = sorted(input_df["date"].unique())

        # 评估指标
        metrics = []

        # 滚动窗口遍历
        result_dfs = []
        for i in range(window, len(sorted_dates)):
            # ====================
            # 样本内训练
            # ====================
            # -1 获取训练窗口数据
            train_window = sorted_dates[i - window: i]
            # -2 获取训练数据
            x_train, y_train = (
                input_df.loc[input_df["date"].isin(train_window), x_cols],
                input_df.loc[input_df["date"].isin(train_window), y_col]
            )
            # -3 过采样（平衡训练集）
            x_train, y_train = self.utils.feature.over_sampling(x_train, y_train)
            # -4 标签转换 离散字符 -> 连续整数
            y_train = pd.Series(
                self.le.fit_transform(y_train),
                index=y_train.index
            )
            # -5 训练模型
            model = XGBClassifier(objective='multi:softmax')
            model.fit(
                x_train,
                y_train,
            )

            # ====================
            # 样本外预测
            # ====================
            # -1 获取预测日数据
            predict_date = sorted_dates[i]
            # -2 获取测试窗口数据
            true_df = input_df.loc[input_df["date"] == predict_date]
            x_true, y_true = (
                input_df.loc[input_df["date"] == predict_date, x_cols],
                input_df.loc[input_df["date"] == predict_date, y_col]
            )
            # -3 模型预测
            y_predict = model.predict(x_true)
            # -4 标签转换 连续整数 -> 离散字符
            true_df["group"] = pd.Series(
                self.le.inverse_transform(y_predict),
                index=true_df.index
            )

            # ====================
            # 归因分析（多分类）
            # ====================
            shap_df = self.shap_for_multiclass(
                model,
                factors_name=x_cols,
                x_true=x_true,
                y_true=self.le.transform(y_true),
                y_predict=y_predict,
            )
            true_df = pd.concat([true_df, shap_df], axis=1)

            # ====================
            # 数据整合（原值、预测收益率/分组、仓位权重、实际股数）
            # ====================
            result_dfs.append(true_df)

            # ====================
            # 模型评估
            # ====================
            try:
                metrics_series = self.calculate_classification_metrics(y_true, true_df["group"])
                metrics_series.name = predict_date
                metrics.append(metrics_series)
            except ValueError:
                continue

        return (pd.concat(result_dfs, ignore_index=True),
                pd.concat(metrics, axis=1).mean(axis=1).to_frame(name="value").T)

    def run(
            self
    ) -> dict[str, pd.DataFrame]:
        """
        线性模型处理流程：
            -1 因子数值处理
            -2 因子升维/降维
            -3 模型训练、预测、分组
        """
        # ----------------------------------
        # 数值处理
        # ----------------------------------
        # 数据预处理
        self.input_df = self._pre_processing(self.input_df)
        # 标签预设
        self.input_df = self.processor.classification.divide_into_group(
            self.input_df,
            processed_factor_col="pctChg",
            group_mode=self.model_setting.group_mode,
            group_nums=self.model_setting.group_nums,
            group_label=self.model_setting.group_label,
            group_col="origin_group"
        )

        # ----------------------------------
        # 因子降维
        # ----------------------------------
        level_2_df = self._factors_synthesis(
            self.input_df,
            mode="THREE_TO_TWO"
        )

        # ----------------------------------
        # 因子相关性
        # ----------------------------------
        corr_df = self.calculate_factors_corr(
            factors_df=level_2_df,
            factors_name=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist()
        )

        # ----------------------------------
        # 模型
        # ----------------------------------
        pred_df, estimate_metric = self.model_training_and_predict(
            input_df=level_2_df,
            x_cols=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist(),
            y_col="origin_group",
            window=self.model_setting.factor_weight_window
        )

        return {
            "模型": pred_df,
            "模型评估": estimate_metric,
            "因子相关性": corr_df,
            "因子shap值": pred_df.filter(like='shap_').abs().mean().sort_values(ascending=False)
        }


########################################################################
class XGBoostClassificationNSModel(ModelTemplate):
    """XGBoost分类模型（非标）"""

    def __init__(
            self,
            input_df: pd.DataFrame,
            model_setting: dataclass,
            descriptive_factors: list[str],
            index_data: dict[str, pd.DataFrame] | None = None,
    ):
        """
        :param input_df: 数据
        :param model_setting: 模型设置
        :param descriptive_factors: 描述性统计因子
        :param index_data: 指数数据
        """
        super().__init__(
            input_df,
            model_setting,
            descriptive_factors
        )
        self.index_data = index_data

        # 分组实参
        self.group_condition = [
            lambda df: df["pctChg"] < df["pctChg"].quantile(0.05),
            lambda df: ((df["pctChg"] <= df["pctChg"].quantile(0.95)) &
                        (df["pctChg"] >= df["pctChg"].quantile(0.05))),
            lambda df: df["pctChg"] > df["pctChg"].quantile(0.95),
        ]
        self.le = LabelEncoder()                                                # 标签转换实例

        # 超参数网格
        # n_estimators/learning_rate
        # max_depth/min_child_weight/gamma
        # reg_alpha/reg_lambda
        # subsample/colsample_bytree
        self.model_param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [10, 50, 100, 500, 1000, 3000, 5000]
        }

    def _pre_processing(
            self,
            input_df: pd.DataFrame,
            prefix: str = "processed"
    ) -> pd.DataFrame:
        """
        数据预处理
            -1 缩尾
            -2 标准化
            -3 中性化
            -4 缩尾
            -5 标准化
        :param input_df: 初始数据
        :param prefix: 预处理生成因子前缀
        :return: 处理过的数据
        """

        def __process_single_date(
                df_: pd.DataFrame,
        ) -> pd.DataFrame:
            """单日数据处理"""
            for setting in self.factors_setting:
                factor_name = setting.factor_name
                processed_col = f"{prefix}_{factor_name}"

                df_[processed_col] = df_[factor_name].copy()
                # -1 第一次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

                # -2 中性化
                if setting.market_value_neutral:
                    df_[processed_col] = self.processor.neutralization.log_market_cap(
                        df_[processed_col],
                        df_["对数市值"],
                        winsorizer=self.processor.winsorizer.percentile,
                        dimensionless=self.processor.dimensionless.standardization
                    )
                if setting.industry_neutral:
                    df_[processed_col] = self.processor.neutralization.industry(
                        df_[processed_col],
                        df_["行业"]
                    )

                # -3 第二次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

            return df_

        result_dfs = []
        for date, group_df in input_df.groupby("date"):
            try:
                processed_df = __process_single_date(group_df)
                result_dfs.append(processed_df)
            except ValueError:
                continue

        # 合并处理结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    def model_training_and_predict(
            self,
            input_df: pd.DataFrame,
            x_cols: list[str],
            y_col: str,
            window: int = 12
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        模型训练与预测
        :param input_df: 输入数据
        :param x_cols: 因子列名
        :param y_col: 目标列名
        :param window: 滚动窗口长度
        :return: 预期收益率
        """
        """
        测试下过采样
        以及 分类加权 哪种效果好
        """
        # 按日期排序并转换为列表
        sorted_dates = sorted(input_df["date"].unique())

        # 评估指标
        metrics = []

        # 滚动窗口遍历
        result_dfs = []
        for i in range(window, len(sorted_dates)):
            # ====================
            # 样本内训练
            # ====================
            # -1 获取训练窗口数据
            train_window = sorted_dates[i - window: i]
            # -2 获取训练数据
            x_train, y_train = (
                input_df.loc[input_df["date"].isin(train_window), x_cols],
                input_df.loc[input_df["date"].isin(train_window), y_col]
            )
            # -3 过采样（平衡训练集）
            x_train, y_train = self.utils.feature.over_sampling(x_train, y_train)
            # -4 标签转换 离散字符 -> 连续整数
            y_train = pd.Series(
                self.le.fit_transform(y_train),
                index=y_train.index
            )
            # -5 训练模型
            model = XGBClassifier(objective='multi:softmax')
            model.fit(
                x_train,
                y_train,
            )

            # ====================
            # 样本外预测
            # ====================
            # -1 获取预测日数据
            predict_date = sorted_dates[i]
            # -2 获取测试窗口数据
            true_df = input_df.loc[input_df["date"] == predict_date]
            x_true, y_true = (
                input_df.loc[input_df["date"] == predict_date, x_cols],
                input_df.loc[input_df["date"] == predict_date, y_col]
            )
            # -3 模型预测
            y_predict = model.predict(x_true)
            # -4 标签转换 连续整数 -> 离散字符
            true_df["group"] = pd.Series(
                self.le.inverse_transform(y_predict),
                index=true_df.index
            )

            # ====================
            # 归因分析（多分类）
            # ====================
            shap_df = self.shap_for_multiclass(
                model,
                factors_name=x_cols,
                x_true=x_true,
                y_true=self.le.transform(y_true),
                y_predict=y_predict,
            )
            true_df = pd.concat([true_df, shap_df], axis=1)

            # ====================
            # 数据整合（原值、预测收益率/分组、仓位权重、实际股数）
            # ====================
            result_dfs.append(true_df)

            # ====================
            # 模型评估
            # ====================
            try:
                metrics_series = self.calculate_classification_metrics(y_true, true_df["group"])
                metrics_series.name = predict_date
                metrics.append(metrics_series)
            except ValueError:
                continue

        return (pd.concat(result_dfs, ignore_index=True),
                pd.concat(metrics, axis=1).mean(axis=1).to_frame(name="value").T)

    def run(
            self
    ) -> dict[str, pd.DataFrame]:
        """
        线性模型处理流程：
            -1 因子数值处理
            -2 因子升维/降维
            -3 模型训练、预测、分组
        """
        # ----------------------------------
        # 数值处理
        # ----------------------------------
        # 标签预设
        self.input_df = self.processor.classification.divide_into_group(
            self.input_df,
            processed_factor_col="pctChg",
            group_mode=self.model_setting.group_mode,
            group_nums=self.model_setting.group_nums,
            group_label=self.model_setting.group_label,
            group_col="origin_group"
        )

        self.input_df = self._direction_reverse(self.input_df)
        self.input_df = self._pre_processing(self.input_df)

        # ----------------------------------
        # 因子降维
        # ----------------------------------
        level_2_df = self._factors_synthesis(
            self.input_df,
            mode="THREE_TO_TWO"
        )

        # ----------------------------------
        # 因子相关性
        # ----------------------------------
        corr_df = self.calculate_factors_corr(
            factors_df=level_2_df,
            factors_name=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist()
        )

        # ----------------------------------
        # 模型
        # ----------------------------------
        pred_df, estimate_metric = self.model_training_and_predict(
            input_df=level_2_df,
            x_cols=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist(),
            y_col="origin_group",
            window=self.model_setting.factor_weight_window
        )

        return {
            "模型": pred_df,
            "模型评估": estimate_metric,
            "因子相关性": corr_df,
            "因子shap值": pred_df.filter(like='shap_').abs().mean().sort_values(ascending=False)
        }


########################################################################
class XGBoostClassificationNSCVModel(ModelTemplate):
    """XGBoost分类模型（非标、超参调优）"""

    def __init__(
            self,
            input_df: pd.DataFrame,
            model_setting: dataclass,
            descriptive_factors: list[str],
            index_data: dict[str, pd.DataFrame] | None = None,
    ):
        """
        :param input_df: 数据
        :param model_setting: 模型设置
        :param descriptive_factors: 描述性统计因子
        :param index_data: 指数数据
        """
        super().__init__(
            input_df,
            model_setting,
            descriptive_factors
        )
        self.index_data = index_data

        # 分组实参
        self.group_condition = [
            lambda df: df["pctChg"] < df["pctChg"].quantile(0.05),
            lambda df: ((df["pctChg"] <= df["pctChg"].quantile(0.95)) &
                        (df["pctChg"] >= df["pctChg"].quantile(0.05))),
            lambda df: df["pctChg"] > df["pctChg"].quantile(0.95),
        ]
        self.le = LabelEncoder()                                                # 标签转换实例

        # 超参数网格
        # n_estimators/learning_rate
        # max_depth/min_child_weight/gamma
        # reg_alpha/reg_lambda
        # subsample/colsample_bytree
        self.model_param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [10, 50, 100, 500, 1000, 3000, 5000]
        }

    def _pre_processing(
            self,
            input_df: pd.DataFrame,
            prefix: str = "processed"
    ) -> pd.DataFrame:
        """
        数据预处理
            -1 缩尾
            -2 标准化
            -3 中性化
            -4 缩尾
            -5 标准化
        :param input_df: 初始数据
        :param prefix: 预处理生成因子前缀
        :return: 处理过的数据
        """

        def __process_single_date(
                df_: pd.DataFrame,
        ) -> pd.DataFrame:
            """单日数据处理"""
            for setting in self.factors_setting:
                factor_name = setting.factor_name
                processed_col = f"{prefix}_{factor_name}"

                df_[processed_col] = df_[factor_name].copy()
                # -1 第一次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

                # -2 中性化
                if setting.market_value_neutral:
                    df_[processed_col] = self.processor.neutralization.log_market_cap(
                        df_[processed_col],
                        df_["对数市值"],
                        winsorizer=self.processor.winsorizer.percentile,
                        dimensionless=self.processor.dimensionless.standardization
                    )
                if setting.industry_neutral:
                    df_[processed_col] = self.processor.neutralization.industry(
                        df_[processed_col],
                        df_["行业"]
                    )

                # -3 第二次 去极值、标准化
                df_[processed_col] = self.processor.winsorizer.percentile(df_[processed_col])
                if setting.standardization:
                    df_[processed_col] = self.processor.dimensionless.standardization(df_[processed_col])

            return df_

        result_dfs = []
        for date, group_df in input_df.groupby("date"):
            try:
                processed_df = __process_single_date(group_df)
                result_dfs.append(processed_df)
            except ValueError:
                continue

        # 合并处理结果
        return pd.concat(result_dfs) if result_dfs else pd.DataFrame()

    def model_training_and_predict(
            self,
            input_df: pd.DataFrame,
            x_cols: list[str],
            y_col: str,
            window: int = 12
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        模型训练与预测
        :param input_df: 输入数据
        :param x_cols: 因子列名
        :param y_col: 目标列名
        :param window: 滚动窗口长度
        :return: 预期收益率
        """
        """
        测试下过采样
        以及 分类加权 哪种效果好
        """
        # 按日期排序并转换为列表
        sorted_dates = sorted(input_df["date"].unique())

        # 评估指标
        metrics = []

        # 滚动窗口遍历
        result_dfs = []
        for i in range(window, len(sorted_dates)):
            # ====================
            # 样本内训练
            # ====================
            # -1 获取训练窗口数据
            train_window = sorted_dates[i - window: i]
            # -2 获取训练数据
            x_train, y_train = (
                input_df.loc[input_df["date"].isin(train_window), x_cols],
                input_df.loc[input_df["date"].isin(train_window), y_col]
            )
            # -3 过采样（平衡训练集）
            x_train, y_train = self.utils.feature.over_sampling(x_train, y_train)
            # -4 标签转换 离散字符 -> 连续整数
            y_train = pd.Series(
                self.le.fit_transform(y_train),
                index=y_train.index
            )
            # -5 训练模型
            model = XGBClassifier(objective='multi:softmax')
            model.fit(
                x_train,
                y_train,
            )

            # ====================
            # 样本外预测
            # ====================
            # -1 获取预测日数据
            predict_date = sorted_dates[i]
            # -2 获取测试窗口数据
            true_df = input_df.loc[input_df["date"] == predict_date]
            x_true, y_true = (
                input_df.loc[input_df["date"] == predict_date, x_cols],
                input_df.loc[input_df["date"] == predict_date, y_col]
            )
            # -3 模型预测
            y_predict = model.predict(x_true)
            # -4 标签转换 连续整数 -> 离散字符
            true_df["group"] = pd.Series(
                self.le.inverse_transform(y_predict),
                index=true_df.index
            )

            # ====================
            # 归因分析（多分类）
            # ====================
            shap_df = self.shap_for_multiclass(
                model,
                factors_name=x_cols,
                x_true=x_true,
                y_true=self.le.transform(y_true),
                y_predict=y_predict,
            )
            true_df = pd.concat([true_df, shap_df], axis=1)

            # ====================
            # 数据整合（原值、预测收益率/分组、仓位权重、实际股数）
            # ====================
            result_dfs.append(true_df)

            # ====================
            # 模型评估
            # ====================
            try:
                metrics_series = self.calculate_classification_metrics(y_true, true_df["group"])
                metrics_series.name = predict_date
                metrics.append(metrics_series)
            except ValueError:
                continue

        return (pd.concat(result_dfs, ignore_index=True),
                pd.concat(metrics, axis=1).mean(axis=1).to_frame(name="value").T)

    def run(
            self
    ) -> dict[str, pd.DataFrame]:
        """
        线性模型处理流程：
            -1 因子数值处理
            -2 因子升维/降维
            -3 模型训练、预测、分组
        """
        # ----------------------------------
        # 数值处理
        # ----------------------------------
        # 标签预设
        self.input_df = self.processor.classification.divide_into_group(
            self.input_df,
            processed_factor_col="pctChg",
            group_mode=self.model_setting.group_mode,
            group_nums=self.model_setting.group_nums,
            group_label=self.model_setting.group_label,
            group_col="origin_group"
        )

        self.input_df = self._direction_reverse(self.input_df)
        self.input_df = self._pre_processing(self.input_df)

        # ----------------------------------
        # 因子降维
        # ----------------------------------
        level_2_df = self._factors_synthesis(
            self.input_df,
            mode="THREE_TO_TWO"
        )

        # ----------------------------------
        # 因子相关性
        # ----------------------------------
        corr_df = self.calculate_factors_corr(
            factors_df=level_2_df,
            factors_name=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist()
        )

        # ----------------------------------
        # 模型
        # ----------------------------------
        pred_df, estimate_metric = self.model_training_and_predict(
            input_df=level_2_df,
            x_cols=level_2_df.columns[~level_2_df.columns.isin(self.keep_cols)].tolist(),
            y_col="origin_group",
            window=self.model_setting.factor_weight_window
        )

        return {
            "模型": pred_df,
            "模型评估": estimate_metric,
            "因子相关性": corr_df,
            "因子shap值": pred_df.filter(like='shap_').abs().mean().sort_values(ascending=False)
        }
