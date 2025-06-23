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
"""
    -1 模型集成
    -2 风格因子中性化
    -5 因子跟踪（除 估值价差 之外）
    -6 数据 行业分类表 以及 各类数据的完善
    -7 标的池
    
    -9 样本内外区分                                                      60%
    -3 扩张窗口因子
    -4 因子暴露控制 风险评价模型 !!!!                                      80%
    -13 把以前的换手率计算手续费模块加回
    
    -11 新量价因子 
        -1 TNR = 区间涨跌幅 / 区间累加涨跌幅
        -2 TNR diff
        -3 up prob = up prob + 0.5 * (down prob * 1 - up prob * -1)
           down prob = down prob + 0.5 * (up prob * -1 - down prob * 1)
           u2p = up prob - down prob
           
    -12 策略容量 每只个股可以成交当日成交额的5%，若当日无法完成交易，则再下一日继续尝试
    
"""

"""
商品日内


300多个 降维后40-50个
持股数量 80-100 选股3000+
65% 财务因子
25% 量价
剩余 另类数据因子
80%线性
20%非线性
年换手40-50倍

微盘 + 择时 + 优选
实盘！！！
因子等权 75量价 25基本面 70反转 25-30动量
市场择时 -> 仓位管理 标的池变换
平均市值 65亿元
单体1% 一SW级行业20%
年化双边换手60-80倍
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
    "linearReg": LinearRegressionModel,
    "traditionalLinearReg": LinearRegressionTraditionalModel,
    "linearTestReg": LinearRegressionTestModel,
    "xgboostReg": XGBoostRegressionModel,
    "xgboostCVReg": XGBoostRegressionCVModel,
    "xgboostCla": XGBoostClassificationModel,
    "randomForestReg": RandomForestClassificationModel,
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
    source_dir = "模型因子训练集"
    storage_dir = "最优化测试/linearTestReg"

    filter_mode: FILTER_MODE = "_entire_filter"
    # 模型参数设置
    model_setting = ModelSetting(
        # 模型/周期/因子
        model="linearTestReg",
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
        class_level="一级行业",
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
