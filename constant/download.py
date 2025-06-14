"""
枚举类
映射字典
指数代码
爬虫请求头
"""

from enum import Enum


#############################################################
class AdjustmentMode(Enum):
    BACKWARD = "1"
    FORWARD = "2"
    UNADJUSTED = "3"


#############################################################
class SheetName(Enum):
    BACKWARD = "backward_adjusted"
    FORWARD = "split_adjusted"
    UNADJUSTED = "unadjusted"


#############################################################
# 映射字典 模式：工作表名
SHEET_NAME_MAP: dict[str, str] = {
    mode.value: sheet.value
    for mode, sheet in zip(
        AdjustmentMode,
        SheetName
    )
}


#############################################################
class TimeDimension(Enum):
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    HALF = "half"
    YEAR = "year"


#############################################################
INDUSTRY_TYPE = {
    '证券Ⅱ': '1',
    '保险Ⅱ': '2',
    '城商行Ⅱ': '3', '股份制银行Ⅱ': '3', '国有大型银行Ⅱ': '3', '农商行Ⅱ': '3'
}
