from dataclasses import dataclass


###################################################
@dataclass
class BasicChartSpecs:
    """基础图表的配置结构"""
    title: str                          # 图表标题
    data_source: str                    # 数据来源（对应数据字典的key）
    column: str | list[str]             # 数据列名
    percent: bool = False               # 百分比

    width: str = '1800px'               # 宽度
    height: str = '600px'               # 高度
    orientation: str = ""               # 方向控制（bar）
    date: bool = True                   # 时间序列仅取 date
    font_family: str = 'KaiTi'          # 字体


###################################################
@dataclass
class QuadrantsChartSpecs:
    title: str                                  # 图表标题

    ul_data_source: str = ''                    # 左上数据来源 upper_left
    ul_column: str | list[str] = ''             # 左上数据列名
    ul_chart: str = 'bar'                       # 左上图表类型
    ul_percent: bool = False                    # 左上百分比

    ll_data_source: str = ''                    # 左下数据来源 lower_left
    ll_column: str | list[str] = ''             # 左下数据列名
    ll_chart: str = 'bar'                       # 左下图表类型
    ll_percent: bool = False                    # 左下百分比

    ur_data_source: str = ''                    # 右上数据来源 upper_right
    ur_column: str | list[str] = ''             # 右上数据列名
    ur_chart: str = 'bar'                       # 右上图表类型
    ur_percent: bool = False                    # 右上百分比

    lr_data_source: str = ''                    # 右下数据来源 lower_right
    lr_column: str | list[str] = ''             # 右下数据列名
    lr_chart: str = 'bar'                       # 右下图标类型
    lr_percent: bool = False                    # 右下百分比

    date: bool = True                           # 时间序列仅取 date

    font_family: str = 'KaiTi'                    # 字体
    main_pie_show: bool = False                 # 是否展示主饼图
