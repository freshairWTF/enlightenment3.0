"""
显示导入特定模型类
"""
from linear_regression_model import LinearRegressionModel
from linear_regression_model_traditional import LinearRegressionTraditionalModel
from linear_regression_model_test import LinearRegressionTestModel
from xgboost_regression_cv_model import XGBoostRegressionCVModel
from xgboost_regression_model import XGBoostRegressionModel

from xgboost_classification_model import XGBoostClassificationModel
from xgboost_classification_NS_model import XGBoostClassificationNSModel
from xgboost_classification_CV_model import XGBoostClassificationCVModel
from xgboost_classification_NS_CV_model import XGBoostClassificationNSCVModel

from randomforest_classification_model import RandomForestClassificationModel

from fight_linear_regression_model import FightLinearRegressionModel, FightLinearRegressionHigherModel


__all__ = [
    # 线性
    "LinearRegressionModel",
    "LinearRegressionTraditionalModel",
    "LinearRegressionTestModel",

    "XGBoostRegressionModel",
    "XGBoostRegressionCVModel",

    # 分类
    "XGBoostClassificationModel",
    "XGBoostClassificationNSModel",
    "XGBoostClassificationCVModel",
    "XGBoostClassificationNSCVModel",

    "RandomForestClassificationModel",

    # 实盘
    "FightLinearRegressionModel",
    "FightLinearRegressionHigherModel",
]