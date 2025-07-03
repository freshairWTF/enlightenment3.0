"""实盘信号生成"""

from model.model_service import ModelAnalyzer
from constant.type_ import FILTER_MODE
from constant.fight_model_setting import LINEAR_MODEL_SETTING


MODEL_SETTING = {
    "ZScoreLinearReg": LINEAR_MODEL_SETTING,
}


# --------------------------------------------
def model_backtest(model_setting_):
    """模型回测"""
    analyzer = ModelAnalyzer(
        model=model_setting_.model,
        model_setting=model_setting_,
        filter_mode=filter_mode,
        source_dir=source_dir,
        storage_dir=storage_dir,
        cycle=model_setting_.cycle,
    )
    analyzer.run()


# --------------------------------------------
if __name__ == "__main__":
    # ----------------------------------
    # 参数
    # ----------------------------------
    source_dir = "实盘20250629"                       # 数据路径
    storage_dir = "shap/非线性模型-测试"                # 存储路径
    model_setting = "ZScoreLinearReg"                # 模型选择
    total_capital = 20000000                         # 可配置资金

    # ----------------------------------
    # 配置/生成
    # ----------------------------------
    # 补齐模型配置
    model_setting = MODEL_SETTING[model_setting]
    model_setting.total_capital = total_capital
    # 标的池（白名单）
    filter_mode: FILTER_MODE = "_white_filter"
    # 生成买卖信号
    model_backtest(MODEL_SETTING[model_setting])
