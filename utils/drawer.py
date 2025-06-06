"""绘图"""
from typing import Literal
from pathlib import Path
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, Scatter, Page, Grid, Timeline, HeatMap
from pyecharts.components import Table
from pyecharts.commons.utils import JsCode

import numpy as np
import pandas as pd

from constant.draw_specs import *


###############################################################
# 若pyecharts服务器响应有误，则需要启动本地服务器：python -m http.server
# 以下为静态资源服务器指向重定位，若服务器响应正常，则可关闭
# 配置CurrentConfig.ONLINE_HOST为本机地址资源
from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = "http://127.0.0.1:8000/assets/v5/"


###############################################################
COLOR_MAP = [
    '#FF731D', '#004d61', '#bc8420', '#CF0A0A', '#83FFE6', '#0000A1', '#fff568', '#0080ff', '#7A57D1',
    '#81C6E8', '#385098', '#ffb5ba', '#EA047E', '#B1AFFF', '#425F57', '#CFFF8D', '#100720', '#18978F',
    '#F9CEEE', '#7882A4', '#E900FF', '#84DFFF', '#B2EA70', '#FED2AA', '#49FF00', '#14279B', '#911F27',
    '#00dffc',
]


###############################################################
class Drawer(object):

    BASIC_METHOD = [
        '_basic_line', '_basic_bar', '_basic_scatter', '_basic_heat_map', '_basic_table'
    ]
    QUADRANTS_METHOD = [
        '_quadrants', '_upper_lower_dichotomy'
    ]
    CHART_CONFIG = BasicChartSpecs | QuadrantsChartSpecs

    def __init__(
            self,
            path: Path,
            pages_name: list,
            pages_config: dict[str, CHART_CONFIG],
            data_container: dict,
            layout: Literal['simple', 'draggable'] = 'simple',
    ):
        """
        :param path: 存储路径
        :param pages_name: 页名列表
        :param pages_config: 页面配置
        :param data_container: 数据容器
        :param layout: 布局模式 ('simple' 或 'draggable')
        """
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

        self.pages = {
            name: Page(layout=Page.DraggablePageLayout if layout == 'draggable' else Page.SimplePageLayout)
            for name in pages_name
        }
        self.pages_config = pages_config
        self.data_container = data_container

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def draw(
            self
    ) -> None:
        """绘图接口"""
        for page in self.pages_config.keys():
            page_config = self._get_page_config(page)
            for method_rank, chart_config in page_config.items():
                # 方法名
                method_name = method_rank.split('-')[0]
                # 聚合数据
                data = self._get_agg_data(method_name, chart_config)
                if method_name and hasattr(self, method_name):
                    self.pages[page].add(
                        getattr(self, method_name)(chart_config, data)
                    )
                else:
                    raise ValueError(f'未实现的方法: {method_name}')

    def render(
            self
    ) -> None:
        """渲染所有页面"""
        for name, page in self.pages.items():
            page.render((self.path / name).with_suffix('.html'))

    # --------------------------
    # 其他 私有方法
    # --------------------------
    def _get_page_config(
            self,
            page_name: str
    ) -> dict[str, CHART_CONFIG]:
        """获取指定页面的所有图表配置"""
        return self.pages_config.get(page_name, {})

    @classmethod
    def _get_basic_chart(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame,
            chart_type: Literal['line', 'bar'],
    ) -> Line | Bar:
        if chart_type == 'line':
            return cls._basic_line(config=config, data=data)
        elif chart_type == 'bar':
            return cls._basic_bar(config=config, data=data)
        else:
            raise ValueError(f'不支持该图标类型: {chart_type}')

    # --------------------------
    # 数据类 私有方法
    # --------------------------
    def _get_agg_data(
            self,
            method: str,
            chart_config: CHART_CONFIG
    ) -> pd.DataFrame:
        """获取聚合数据接口"""
        if method in self.BASIC_METHOD:
            return self.__agg_basic_data(chart_config)
        elif method in self.QUADRANTS_METHOD:
            return self.__agg_quadrants_data(chart_config)
        else:
            print(f'绘图类暂不支持该方法: {method}')
            return pd.DataFrame()

    def __agg_basic_data(
            self,
            config: BasicChartSpecs
    ) -> pd.DataFrame:
        """聚合基础图表数据"""
        if config.column:
            df = self.data_container[config.data_source][config.column].dropna(how='all')
        else:
            df = self.data_container[config.data_source].dropna(how='all')

        if df.empty:
            return pd.DataFrame()

        if config.date:
            return df.reindex(df.index.date)
        return df

    def __agg_quadrants_data(
            self,
            config: QuadrantsChartSpecs
    ) -> pd.DataFrame:
        """聚合四分高级图表数据"""
        result = {}
        index = None

        for direct in ['ul', 'ur', 'll', 'lr']:
            data_source = getattr(config, f'{direct}_data_source', '')
            column = getattr(config, f'{direct}_column', '')
            if data_source and column:
                series = self.data_container[data_source][column]
                # 首次遇到有效数据时设置基准索引
                if index is None:
                    index = series.index
                # 重新索引对齐
                result[direct] = series.reindex(index)

        # 数据聚合
        result = pd.concat(result.values(), axis=1, keys=result.keys()).dropna(how='all')

        # 时间索引
        if config.date:
            return result.reindex(result.index.date)
        return result

    # --------------------------
    # 基础图表方法
    # --------------------------
    @classmethod
    def _basic_line(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame,
    ) -> Line:
        """基础折线图"""
        # 图表
        line = (
            Line(init_opts=opts.InitOpts(width='1800px'))
            .add_xaxis(data.index.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=config.title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family)
                ),
                datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
            )
        )
        # 添加数据
        for legend, series in data.items():
            line.add_yaxis(
                series_name=legend,
                y_axis=series.tolist(),
                label_opts=opts.LabelOpts(is_show=False)
            )

        return line

    @classmethod
    def _basic_bar(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame,
    ) -> Bar:
        """基础柱状图"""
        # 图表
        bar = (
            Bar(init_opts=opts.InitOpts(width='1800px'))
            .add_xaxis(data.index.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=config.title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family)
                ),
                datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
            )
        )
        # 添加数据
        for legend, series in data.items():
            bar.add_yaxis(
                series_name=legend,
                y_axis=series.tolist(),
                category_gap='80%',
                label_opts=opts.LabelOpts(is_show=False)
            )

        return bar

    @classmethod
    def _basic_table(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame
    ) -> Table:
        """基础表格"""
        headers = [data.index.name or ""] + data.columns.tolist()
        rows = []
        for idx, row in data.iterrows():
            rows.append([str(idx)] + [f"{x:.2f}" if isinstance(x, (int, float)) else str(x) for x in row.values])

        return (
            Table()
            .add(headers, rows)
            .set_global_opts(title_opts=opts.ComponentTitleOpts(title=config.title))
        )

    @classmethod
    def _basic_scatter(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame
    ) -> Scatter:
        """基础散点图"""
        scatter_data = []
        for col in data.columns:
            scatter_data.extend(
                [opts.ScatterItem(
                    name=category,
                    value=[
                        x_idx,
                        float(val) if not pd.isna(val) else 0.0
                    ],
                    symbol_size=8
                ) for (category, val), x_idx in zip(
                    data[col].items(),
                    range(len(data.index))
                )]
            )

        return (
            Scatter(init_opts=opts.InitOpts(width='1800px', height='600px'))
            .add_xaxis(xaxis_data=data.index.tolist())
            .add_yaxis(
                series_name=config.title,
                y_axis=scatter_data,
                symbol_size=8,
                label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=config.title),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        "function(params) {"
                        "return params.value[0] + ': ' + params.value[1];"
                        "}")
                )
            )
        )

    @classmethod
    def _basic_heat_map(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame
    ) -> HeatMap:
        """基本热力图"""
        # 处理数据索引和列名
        data = data.copy()
        data.index = data.index.astype(str)
        data.columns = data.columns.astype(str)

        # 生成热力图数据
        heatmap_data = []
        for idx in data.index:
            for col in data.columns:
                heatmap_data.append([
                    col, idx, data.loc[idx, col]
                ])

        return (
            HeatMap(init_opts=opts.InitOpts(width='1800px', height='800px'))
            .add_xaxis(data.columns.tolist())
            .add_yaxis(
                series_name=config.title,
                yaxis_data=data.index.tolist(),
                value=heatmap_data,
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="inside",
                    color="black",
                    font_size=12,
                    formatter=JsCode(
                        "function(params) {"
                        "  if (params.value[2]) {"  
                        "    return params.value[2].toFixed(2);"  
                        "  } else {"
                        "    return '';"  
                        "  }"
                        "}"
                    )
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=config.title),
                visualmap_opts=opts.VisualMapOpts(
                    min_=np.trunc(data.values.min())-1,
                    max_=np.trunc(data.values.max()),
                    is_calculable=True,
                    pos_left="30px"
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        "function(params) {"
                        "return params.value[0] + ' - ' + params.value[1] + ': ' + params.value[2];"
                        "}")
                )
            )
        )

    # --------------------------
    # 高级图表方法
    # --------------------------
    @classmethod
    def _quadrants(
            cls,
            config: QuadrantsChartSpecs,
            data: pd.DataFrame
    ) -> Grid:
        """四象限布局（柱图、折线）"""
        grid = Grid(init_opts=opts.InitOpts(width='1800px', height='900px'))

        # 坐标系参数配置
        layout_config = {
            "ul": {  # 左上
                "grid_opts": opts.GridOpts(pos_right="55%", pos_bottom="55%"),
                "title_pos": {"pos_left": "10%"},
                "legend_opts": opts.LegendOpts(pos_left="15%", pos_top="3%")
            },
            "ll": {  # 左下
                "grid_opts": opts.GridOpts(pos_right="55%", pos_top="55%"),
                "title_pos": {"pos_left": "10%", "pos_bottom": "50%"},
                "legend_opts": opts.LegendOpts(pos_left="15%", pos_bottom='47%')
            },
            "ur": {  # 右上
                "grid_opts": opts.GridOpts(pos_left="55%", pos_bottom="55%"),
                "title_pos": {"pos_right": "10%"},
                "legend_opts": opts.LegendOpts(pos_left='60%', pos_top="3%")
            },
            "lr": {  # 右下
                "grid_opts": opts.GridOpts(pos_left="55%", pos_top="55%"),
                "title_pos": {"pos_right": "10%", "pos_bottom": "50%"},
                "legend_opts": opts.LegendOpts(pos_left='60%', pos_bottom='47%')
            }
        }

        for position in ["ul", "ll", "ur", "lr"]:
            if position in data.columns:
                # 删除最上层的方位index
                pos_data = data[[position]].droplevel(0, axis=1)
                chart = cls._get_basic_chart(
                    data=pos_data,
                    config=config,
                    chart_type=getattr(config, f'{position}_chart')
                )
                # 动态构建标题选项（合并位置参数和动态标题）
                dynamic_title = getattr(config, f'{position}_column')
                title_opts = opts.TitleOpts(
                    title=dynamic_title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family),
                    **layout_config[position]["title_pos"]
                )
                # 应用布局配置
                chart.set_global_opts(
                    title_opts=title_opts,
                    legend_opts=layout_config[position]["legend_opts"],
                    datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                    tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
                )
                # 添加到网格
                grid.add(
                    chart,
                    grid_opts=layout_config[position]["grid_opts"]
                )

        return grid

    @classmethod
    def _upper_lower_dichotomy(
            cls,
            config: QuadrantsChartSpecs,
            data: pd.DataFrame
    ) -> Grid:
        """上下布局（柱图、折线）"""
        grid = Grid(init_opts=opts.InitOpts(width='1800px', height='900px'))

        # 坐标系参数配置
        layout_config = {
            "ul": {  # 上
                "grid_opts": opts.GridOpts(pos_bottom="55%"),
                "title_pos": {},
                "legend_opts": opts.LegendOpts()
            },
            "ll": {  # 下
                "grid_opts": opts.GridOpts(pos_top="55%"),
                "title_pos": {"pos_bottom": "47%"},
                "legend_opts": opts.LegendOpts(pos_bottom='47%')
            },
        }

        for position in ["ul", "ll"]:
            if position in data.columns:
                # 删除最上层的方位index
                pos_data = data[[position]].droplevel(0, axis=1)
                chart = cls._get_basic_chart(
                    data=pos_data,
                    config=config,
                    chart_type=getattr(config, f'{position}_chart')
                )
                # 动态构建标题选项（合并位置参数和动态标题）
                dynamic_title = getattr(config, f'{position}_column')
                title_opts = opts.TitleOpts(
                    title=dynamic_title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family),
                    **layout_config[position]["title_pos"]
                )
                # 应用布局配置
                chart.set_global_opts(
                    title_opts=title_opts,
                    legend_opts=layout_config[position]["legend_opts"],
                    datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                    tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
                )
                # 添加到网格
                grid.add(
                    chart,
                    grid_opts=layout_config[position]["grid_opts"]
                )

        return grid


###############################################################
class IndividualDrawer(object):

    BASIC_METHOD = [
        '_basic_line', '_basic_bar', '_basic_scatter', '_basic_heat_map', '_basic_table'
    ]
    QUADRANTS_METHOD = [
        '_quadrants', '_blpp', '_upper_lower_dichotomy', '_overlap'
    ]
    CHART_CONFIG = BasicChartSpecs | QuadrantsChartSpecs

    def __init__(
            self,
            path: Path,
            pages_name: list,
            pages_config: dict[str, CHART_CONFIG],
            data_container: dict,
            layout: Literal['simple', 'draggable'] = 'simple',
    ):
        """
        :param path: 存储路径
        :param pages_name: 页名列表
        :param pages_config: 页面配置
        :param data_container: 数据容器
        :param layout: 布局模式 ('simple' 或 'draggable')
        """
        self.path = path
        self.pages = {
            name: Page(layout=Page.DraggablePageLayout if layout == 'draggable' else Page.SimplePageLayout)
            for name in pages_name
        }
        self.pages_config = pages_config
        self.data_container = data_container

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def draw(
            self
    ) -> None:
        """绘图接口"""
        for page in self.pages_config.keys():
            page_config = self._get_page_config(page)
            for method_rank, chart_config in page_config.items():
                # 方法名
                method_name = method_rank.split('-')[0]
                # 聚合数据
                data = self._get_agg_data(method_name, chart_config)
                if method_name and hasattr(self, method_name):
                    self.pages[page].add(
                        getattr(self, method_name)(chart_config, data)
                    )
                else:
                    raise ValueError(f'未实现的方法: {method_name}')

    def render(
            self
    ) -> None:
        """渲染所有页面"""
        for name, page in self.pages.items():
            page.render((self.path / name).with_suffix('.html'))

    # --------------------------
    # 其他 私有方法
    # --------------------------
    def _get_page_config(
            self,
            page_name: str
    ) -> dict[str, CHART_CONFIG]:
        """获取指定页面的所有图表配置"""
        return self.pages_config.get(page_name, {})

    @classmethod
    def _get_basic_chart(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame,
            chart_type: Literal['line', 'bar'],
    ) -> Line | Bar:
        if chart_type == 'line':
            return cls._basic_line(config=config, data=data)
        elif chart_type == 'bar':
            return cls._basic_bar(config=config, data=data)
        else:
            raise ValueError(f'不支持该图标类型: {chart_type}')

    # --------------------------
    # 数据类 私有方法
    # --------------------------
    def _get_agg_data(
            self,
            method: str,
            chart_config: CHART_CONFIG
    ) -> pd.DataFrame:
        """获取聚合数据接口"""
        if method in self.BASIC_METHOD:
            return self.__agg_basic_data(chart_config)
        elif method in self.QUADRANTS_METHOD:
            return self.__agg_quadrants_data(chart_config)
        else:
            print(f'绘图类暂不支持该方法: {method}')
            return pd.DataFrame()

    def __agg_basic_data(
            self,
            config: BasicChartSpecs
    ) -> pd.DataFrame:
        """聚合基础图表数据"""
        data = self.data_container[config.data_source][[config.column]].dropna(how='all')
        if config.date:
            return data.reindex(data.index.date)
        return data

    def __agg_quadrants_data(
            self,
            config: QuadrantsChartSpecs
    ) -> pd.DataFrame:
        """聚合四分高级图表数据"""
        result = {}
        index = None

        for direct in ['ul', 'ur', 'll', 'lr']:
            data_source = getattr(config, f'{direct}_data_source', '')
            column = getattr(config, f'{direct}_column', '')
            if data_source and column:
                if isinstance(column, str):
                    df = self.data_container[data_source][[column]]
                else:
                    df = self.data_container[data_source][column]
                # 首次遇到有效数据时设置基准索引
                if index is None:
                    index = df.index
                # 重新索引对齐
                result[direct] = df.reindex(index)

        # 数据聚合
        result = pd.concat(result.values(), axis=1, keys=result.keys()).dropna(how='all')
        # 时间索引
        if config.date:
            return result.reindex(result.index.date)
        return result

    # --------------------------
    # 基础图表方法
    # --------------------------
    @classmethod
    def _basic_line(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame,
    ) -> Line:
        """基础折线图"""
        # 图表
        line = (
            Line(init_opts=opts.InitOpts(width='1800px'))
            .add_xaxis(data.index.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=config.title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family)
                ),
                datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
            )
        )
        # 添加数据
        for legend, series in data.items():
            line.add_yaxis(
                series_name=legend,
                y_axis=series.tolist(),
                label_opts=opts.LabelOpts(is_show=False)
            )

        return line

    @classmethod
    def _basic_bar(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame,
    ) -> Bar:
        """基础柱状图"""
        # 图表
        bar = (
            Bar(init_opts=opts.InitOpts(width='1800px'))
            .add_xaxis(data.index.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=config.title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family)
                ),
                datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
            )
        )
        # 添加数据
        for legend, series in data.items():
            bar.add_yaxis(
                series_name=legend,
                y_axis=series.tolist(),
                # stack='stack1' if stack else None,
                category_gap='80%',
                label_opts=opts.LabelOpts(is_show=False)
            )

        return bar

    @classmethod
    def _basic_table(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame
    ) -> Table:
        """基础表格"""
        headers = [data.index.name or ""] + data.columns.tolist()
        rows = []
        for idx, row in data.iterrows():
            rows.append([str(idx)] + [f"{x:.2f}" if isinstance(x, (int, float)) else str(x) for x in row.values])

        return (
            Table()
            .add(headers, rows)
            .set_global_opts(title_opts=opts.ComponentTitleOpts(title=config.title))
        )

    @classmethod
    def _basic_scatter(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame
    ) -> Scatter:
        """基础散点图"""
        scatter_data = []
        for col in data.columns:
            scatter_data.extend(
                [opts.ScatterItem(
                    name=str(idx),
                    value=[idx, val],
                    symbol_size=8
                ) for idx, val in data[col].items()]
            )

        return (
            Scatter(init_opts=opts.InitOpts(width='1800px', height='600px'))
            .add_xaxis(xaxis_data=data.index.tolist())
            .add_yaxis(
                series_name=config.title,
                y_axis=scatter_data,
                symbol_size=8,
                label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=config.title),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        "function(params) {"
                        "return params.value[0] + ': ' + params.value[1];"
                        "}")
                )
            )
        )

    @classmethod
    def _basic_heat_map(
            cls,
            config: CHART_CONFIG,
            data: pd.DataFrame
    ) -> HeatMap:
        """基本热力图"""
        # 处理数据索引和列名
        data = data.copy()
        data.index = data.index.astype(str)
        data.columns = data.columns.astype(str)

        # 生成热力图数据
        heatmap_data = []
        for idx in data.index:
            for col in data.columns:
                heatmap_data.append([col, idx, data.loc[idx, col]])

        return (
            HeatMap(init_opts=opts.InitOpts(width='1800px', height='800px'))
            .add_xaxis(data.columns.tolist())
            .add_yaxis(
                series_name=config.title,
                yaxis_data=data.index.tolist(),
                value=heatmap_data,
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="inside",
                    color="black",
                    font_size=12,
                    formatter=JsCode(
                        "function(params) {"
                        "  if (params.value[2]) {"  
                        "    return params.value[2].toFixed(2);"  
                        "  } else {"
                        "    return '';"  
                        "  }"
                        "}"
                    )
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=config.title),
                visualmap_opts=opts.VisualMapOpts(
                    min_=np.trunc(data.values.min())-1,
                    max_=np.trunc(data.values.max()),
                    is_calculable=True,
                    pos_left="30px"
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        "function(params) {"
                        "return params.value[0] + ' - ' + params.value[1] + ': ' + params.value[2];"
                        "}")
                )
            )
        )

    # --------------------------
    # 高级图表方法
    # --------------------------
    @classmethod
    def _quadrants(
            cls,
            config: QuadrantsChartSpecs,
            data: pd.DataFrame
    ) -> Grid:
        """四象限布局（柱图、折线）"""
        grid = (
            Grid(init_opts=opts.InitOpts(width='1800px', height='900px'))
        )

        # 坐标系参数配置
        layout_config = {
            "ul": {  # 左上
                "grid_opts": opts.GridOpts(pos_right="55%", pos_bottom="55%"),
                "title_pos": {"pos_left": "0%"},
                "legend_opts": opts.LegendOpts(pos_left="15%", pos_top="3%")
            },
            "ll": {  # 左下
                "grid_opts": opts.GridOpts(pos_right="55%", pos_top="55%"),
                "title_pos": {"pos_left": "10%", "pos_bottom": "50%"},
                "legend_opts": opts.LegendOpts(pos_left="15%", pos_bottom='47%')
            },
            "ur": {  # 右上
                "grid_opts": opts.GridOpts(pos_left="55%", pos_bottom="55%"),
                "title_pos": {"pos_right": "10%"},
                "legend_opts": opts.LegendOpts(pos_left='60%', pos_top="3%")
            },
            "lr": {  # 右下
                "grid_opts": opts.GridOpts(pos_left="55%", pos_top="55%"),
                "title_pos": {"pos_right": "10%", "pos_bottom": "50%"},
                "legend_opts": opts.LegendOpts(pos_left='60%', pos_bottom='47%')
            }
        }

        for position in ["ul", "ll", "ur", "lr"]:
            if position in data.columns:
                pos_data = data[[position]].droplevel(0, axis=1)
                chart = cls._get_basic_chart(
                    data=pos_data,
                    config=config,
                    chart_type=getattr(config, f'{position}_chart')
                )
                # 动态构建标题选项（合并位置参数和动态标题）
                title = config.title if position == 'ul' else ''
                title_opts = opts.TitleOpts(
                    title=title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family),
                    **layout_config[position]["title_pos"]
                )
                # 应用布局配置
                chart.set_global_opts(
                    title_opts=title_opts,
                    legend_opts=layout_config[position]["legend_opts"],
                    datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                    tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
                )
                # 添加到网格
                grid.add(
                    chart,
                    grid_opts=layout_config[position]["grid_opts"]
                )

        return grid

    @classmethod
    def _upper_lower_dichotomy(
            cls,
            config: QuadrantsChartSpecs,
            data: pd.DataFrame
    ) -> Grid:
        """上下布局（柱图、折线）"""
        grid = Grid(init_opts=opts.InitOpts(width='1800px', height='900px'))

        # 坐标系参数配置
        layout_config = {
            "ul": {  # 上
                "grid_opts": opts.GridOpts(pos_bottom="55%"),
                "title_pos": {},
                "legend_opts": opts.LegendOpts()
            },
            "ll": {  # 下
                "grid_opts": opts.GridOpts(pos_top="55%"),
                "title_pos": {"pos_bottom": "47%"},
                "legend_opts": opts.LegendOpts(pos_bottom='47%')
            },
        }

        for position in ["ul", "ll"]:
            if position in data.columns:
                pos_data = data[[position]].droplevel(0, axis=1)
                chart = cls._get_basic_chart(
                    data=pos_data,
                    config=config,
                    chart_type=getattr(config, f'{position}_chart')
                )
                # 动态构建标题选项（合并位置参数和动态标题）
                title = config.title if position == 'ul' else ''
                title_opts = opts.TitleOpts(
                    title=title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family),
                    **layout_config[position]["title_pos"]
                )
                # 应用布局配置
                chart.set_global_opts(
                    title_opts=title_opts,
                    legend_opts=layout_config[position]["legend_opts"],
                    datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                    tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
                )
                # 添加到网格
                grid.add(
                    chart,
                    grid_opts=layout_config[position]["grid_opts"]
                )

        return grid

    @classmethod
    def _blpp(
            cls,
            config: QuadrantsChartSpecs,
            data: pd.DataFrame
    ) -> Timeline:
        """BLPP 布局（时间轴、柱图、折线、饼图）"""
        def get_pie_data(series) -> list:
            sum_val = series.sum()
            pie_data_ = []

            if sum_val == 0 or pd.isna(sum_val):
                return pie_data_

            for index, value in series.items():
                ratio = round(value / sum_val * 100, 2)
                if ratio:
                    pie_data_.append(
                        opts.PieItem(name=index, value=ratio)
                    )

            return pie_data_

        # 初始化时间轴
        tl = Timeline(init_opts=opts.InitOpts(width='1800px', height='800px')) \
            .add_schema(pos_left='43%', play_interval=500)

        # 提取数据
        ul_df = data.xs('ul', level=0, axis=1) if 'ul' in data.columns.levels[0] else pd.DataFrame()
        sub_pie_data = {
            'ur': data.xs('ur', level=0, axis=1) if 'ur' in data.columns.levels[0] else pd.DataFrame(),
            'lr': data.xs('lr', level=0, axis=1) if 'lr' in data.columns.levels[0] else pd.DataFrame()
        }

        # 主、副饼图控制参数
        main_pie_show = getattr(config, 'main_pie_show', True)
        sub_pie_show = False if sub_pie_data['ur'].empty and sub_pie_data['lr'].empty else True

        # 坐标系参数配置
        layout_config = {
            "ul": {  # 左上
                "grid_opts": opts.GridOpts(pos_right="55%", pos_bottom="55%"),
                "title_pos": {"pos_left": "0%"},
                "legend_opts": opts.LegendOpts(pos_left="10%", pos_top='2%')
            },
            "ll": {  # 左下
                "grid_opts": opts.GridOpts(pos_right="55%", pos_top="55%"),
                "title_pos": {"pos_left": "10%", "pos_bottom": "50%"},
                "legend_opts": opts.LegendOpts(pos_left="10%", pos_bottom='48%')
            },
            "ur": {  # 右上
                "grid_opts": opts.GridOpts(pos_left="55%", pos_bottom="55%"),
                "title_pos": {"pos_right": "10%"},
                "legend_opts": opts.LegendOpts(pos_left='60%', pos_top="3%")
            },
            "lr": {  # 右下
                "grid_opts": opts.GridOpts(pos_left="55%", pos_top="55%"),
                "title_pos": {"pos_right": "10%", "pos_bottom": "50%"},
                "legend_opts": opts.LegendOpts(pos_left='60%', pos_bottom='47%')
            }
        }
        # -1 兼有主、副饼图
        if main_pie_show and sub_pie_show:
            main_pie_center = "60%"
            sub_pie_center = "85%"
        # -2 仅有 主饼图/副饼图
        else:
            main_pie_center = sub_pie_center = "70%"

        # 遍历时间轴日期
        for date in ul_df.index:
            grid = Grid()

            # -------------------- 左侧 --------------------
            for position in ["ul", "ll"]:
                pos_data = data[[position]].droplevel(0, axis=1)
                chart = cls._get_basic_chart(
                    data=pos_data,
                    config=config,
                    chart_type=getattr(config, f'{position}_chart')
                )
                # 动态构建标题选项（合并位置参数和动态标题）
                title = config.title if position == 'ul' else ''
                title_opts = opts.TitleOpts(
                    title=title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family),
                    **layout_config[position]["title_pos"]
                )
                # 应用布局配置
                chart.set_global_opts(
                    title_opts=title_opts,
                    legend_opts=layout_config[position]["legend_opts"],
                    datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                    tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
                )
                # 添加到网格
                grid.add(
                    chart,
                    grid_opts=layout_config[position]["grid_opts"]
                )

            # -------------------- 右侧 --------------------
            # 主饼图 数据源于 ul_df
            if main_pie_show and not ul_df.empty:
                pie_data = get_pie_data(ul_df.loc[date])
                if pie_data:
                    pie = (
                        Pie()
                        .add(
                            "",
                            pie_data,
                            center=[main_pie_center, "50%"],
                            radius=["30%", "50%"]
                        )
                        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}%"))
                    )
                    grid.add(pie, grid_opts=opts.GridOpts(pos_left="60%"))

            # 副饼图
            for i, position in enumerate(["ur", "lr"]):
                sub_df = sub_pie_data[position]
                if not sub_df.empty:
                    pie_data = get_pie_data(sub_df.loc[date])
                    if pie_data:
                        sub_pie = (
                            Pie()
                            .add(
                                "",
                                pie_data,
                                center=[sub_pie_center, f"{30 + i * 40}%"],
                                radius=["20%", "30%"]
                            )
                            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}%"))
                        )
                        grid.add(sub_pie, grid_opts=layout_config[position]["grid_opts"])

            tl.add(grid, time_point=str(date))

        return tl

    @classmethod
    def _overlap(
            cls,
            config: QuadrantsChartSpecs,
            data: pd.DataFrame
    ) -> Bar:
        """柱线混合图表（支持双Y轴）"""
        # 提取数据
        bar_df = data.xs('ul', level=0, axis=1) if 'ul' in data.columns.levels[0] else pd.DataFrame()
        line_df = data.xs('ll', level=0, axis=1) if 'll' in data.columns.levels[0] else pd.DataFrame()

        # 初始化柱状图
        bar = (
            Bar(init_opts=opts.InitOpts(width='1800px'))
            .add_xaxis(xaxis_data=bar_df.index.tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=config.title,
                    title_textstyle_opts=opts.TextStyleOpts(font_family=config.font_family)
                ),
                datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100, type_='inside'),
                tooltip_opts=opts.TooltipOpts(trigger='axis', trigger_on="mousemove"),
            )
            .extend_axis(
                yaxis=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(formatter="{value}%")
                )
            )
        )

        # 添加柱状图序列
        for name, series in bar_df.items():
            bar.add_yaxis(
                series_name=name,
                y_axis=series.tolist(),
                # stack="stack1" if stack else None,
                gap="50%",
                label_opts=opts.LabelOpts(is_show=False)
            )

        # 添加平均标记线
        # if mark_line:
        #     bar.set_series_opts(
        #         markline_opts=opts.MarkLineOpts(
        #             data=[opts.MarkLineItem(type_="average", name="平均值")]
        #         )
        #     )
        #

        # 初始化折线图
        line = (
            Line()
            .add_xaxis(xaxis_data=bar_df.index.tolist())
            .set_global_opts(
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}%"))
            )
        )

        # 添加折线序列
        for name, series in line_df.items():
            line.add_yaxis(
                series_name=name,
                y_axis=series.tolist(),
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False)
            )

            # 合并图表
            bar.overlap(line)

        return bar
