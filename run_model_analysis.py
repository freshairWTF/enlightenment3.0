"""
模型回测
"""
"""
    -1 线性回归模型的局限？ -> 非权重方法的分类器
    -2 barra中性化 9大类风格因子

    模型实战
    -1 数据检视 数值类型 非空
    -2 基线模型评分 基线模型 评价指标 分层k折交叉验证
    -3 超参数调试 网格搜索 or 随机搜索 早停 一次一参 按照参数性质，分组调整
    -4 基学习器的选择 以及调参
    -5 特诊工程 均值编码
    -6 集成模型 集成 堆叠
    
    把因子降维放到模型层面，不同模型会有不同的处理需求，如线性模型可以降维、可以因子数不同，而非线性模型不行
    模型各自的拟合、预测方法下放到模型层面，不再由模型父类管理
    
    输入模型的数据 从 dict 变为 df？
    如果ic或ir方向发生变化，则在模型内需要再乘一次 -1！在数据处理阶段就乘以-1，对后续模型的影响？
    如果跟预设的方向 相反，则输出警告
    线性模型才需要乘以 -1 ，分类模型没有这种约束，更加证明要将数据处理放入模型中处理
    那么模型层面就独立拥有 数据处理 模型拟合、预测 的方法
    
    如果逐一进行超参的调试，那么只能在最新的预测周期进行
"""
from constant.factor_library import *
from constant.quant_setting import ModelSetting

from model import *
from model.model_service import ModelAnalyzer
from constant.type_ import FILTER_MODE

# -----------------------------
# 模型工厂
# -----------------------------
MODEL = {
    "linear": LinearRegressionModel,
    "traditional_linear": LinearRegressionTraditionalModel,
    "xgboost": XGBoostRegressionModel,
    "randomforest": RandomForestClassificationModel
}


# --------------------------------------------
def model_backtest():
    """模型回测"""
    analyzer = ModelAnalyzer(
        model=MODEL[model_setting.model],
        model_setting=model_setting,
        filter_mode=filter_mode,
        source_dir=source_dir,
        storage_dir=storage_dir,
        cycle=model_setting.cycle,
    )
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    # 路径参数
    source_dir = "20250530-WEEK-跟踪"
    storage_dir = "评估指标调试/traditional_linear-20250530W"

    """
    阿尔法因子   直接ir加权
    弱/风险因子  使用ir衰减
       
    因子暴露控制！！！
    barra风格中性化！！！
    """

    """
    扩展窗口 捕捉长期趋势 ！！！
    很多因子可以试一下
    相较于滚动窗口
    """
    filter_mode: FILTER_MODE = "_entire_filter"

    # 模型参数设置
    model_setting = ModelSetting(
        # 模型/周期/因子
        model="traditional_linear",
        cycle="week",
        factors_setting=list(FACTOR_LIBRARY.values()),
        industry_info={"全部": "三级行业"},

        # 目标因子
        factor_filter=False,
        factor_filter_mode=["_entire_filter"],
        # factor_primary_classification=["基本面因子"],
        # factor_secondary_classification=["质量因子"],
        # factor_half_life=(3, 6),

        # 因子处理方法
        class_level="三级行业",
        bottom_factor_weight_method="ir_decay_weight",
        secondary_factor_weight_method="ir_decay_weight",
        factor_weight_window=12,

        # 分组
        group_nums=20,
        group_mode="frequency",

        # 仓位
        position_weight_method="group_long_only",
        position_distribution=(3, 1),
    )

    # 回测
    model_backtest()
