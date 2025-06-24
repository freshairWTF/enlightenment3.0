from pathlib import Path


BASE_DIR = Path(__file__).parent.parent


class DataPATH:
    FUNDAMENTAL_DATA = BASE_DIR / "fundamental_data" / "cleaned_data"
    BONUS_FINANCING_DATA = BASE_DIR / "fundamental_data" / "cleaned_data" / "bonus_financing"
    FINANCIAL_DATA = BASE_DIR / "fundamental_data" / "cleaned_data" / "financial_data"
    TOTAL_SHARES_DATA = BASE_DIR / "fundamental_data" / "cleaned_data" / "total_shares"
    CIRCULATing_SHARES_DATA = BASE_DIR / "fundamental_data" / "cleaned_data" / "circulating_shares"
    TOP_TEN_CIRCULATING_SHAREHOLDERS = BASE_DIR / "fundamental_data" / "cleaned_data" / "top_ten_circulating_shareholders"
    TOP_TEN_SHAREHOLDERS = BASE_DIR / "fundamental_data" / "cleaned_data" / "top_ten_shareholders"

    KLINE_DATA = BASE_DIR / "kline_data"
    STOCK_KLINE_DATA = BASE_DIR / "kline_data" / "stock"
    ORIGINAL_STOCK_KLINE_DATA = BASE_DIR / "kline_data" / "stock" / "original_day"
    INDEX_KLINE_DATA = BASE_DIR / "kline_data" / "index"
    ORIGINAL_INDEX_KLINE_DATA = BASE_DIR / "kline_data" / "index" / "original_day"
    FUTURE_KLINE_DATA = BASE_DIR / "kline_data" / "future"
    ORIGINAL_FUTURE_KLINE_DATA = BASE_DIR / "kline_data" / "future" / "original_day"

    SUPPORT_DATA = BASE_DIR / "support_file"
    LISTED_NUMS = BASE_DIR / "support_file" / "Listed_Nums"
    TRADING_CALENDAR = BASE_DIR / "support_file" / "Trading_Calendar"
    INDUSTRY_CLASSIFICATION = BASE_DIR / "support_file" / "Industry_Classification_Table"
    INDUSTRY_CLASSIFICATION_UPDATER = BASE_DIR / "support_file" / "Industry_Classification_Updater_Table"

    FINANCIAL_METRICS = BASE_DIR / "analysis" / "financial_metrics"
    KLINE_METRICS = BASE_DIR / "analysis" / "kline_metrics"
    VALUATION_METRICS = BASE_DIR / "analysis" / "valuation_metrics"
    GOVERNANCE_METRICS = BASE_DIR / "analysis" / "governance_metrics"
    STATISTICS_METRICS = BASE_DIR / "analysis" / "statistics_metrics"

    MAC_ANALYSIS_RESULT = BASE_DIR / "result_data" / "分析宏观"
    MESO_ANALYSIS_RESULT = BASE_DIR / "result_data" / "分析中观"
    MICRO_ANALYSIS_RESULT = BASE_DIR / "result_data" / "分析微观"
    INDIVIDUAL_ANALYSIS_RESULT = BASE_DIR / "result_data" / "分析个股"
    QUANT_ANALYSIS_RESULT = BASE_DIR / "result_data" / "分析因子"

    QUANT_CONVERT_RESULT = BASE_DIR / "result_data" / "量化因子"
    QUANT_FACTOR_ANALYSIS_RESULT = BASE_DIR / "result_data" / "量化单因子"
    QUANT_MODEL_ANALYSIS_RESULT = BASE_DIR / "result_data" / "量化模型"

    QUANT_FACTOR_MONITOR_RESULT = BASE_DIR / "result_data" / "监控因子"
    KLINE_MONITOR_RESULT = BASE_DIR / "result_data" / "监控量价"
