import numpy as np
import pandas as pd

from base_metrics import Metrics, depends_on
from constant.type_ import CYCLE, validate_literal_params
from kline_determination import KlineDetermination
from utils.data_processor import DataProcessor

"""
标记半衰期
根据半衰期筛选因子  得到 长、中、短期的选股集合

公司治理指标

行业指标
赫芬达尔指数：测算行业集中度
"""

"""
一、股权集中度类因子
二、股东行为动态因子

三、机构参与度因子
 机构持股集中度 
 计算 ：机构投资者（公募、私募、外资）合计持股比例
。
 有效性 ：光大证券构建的持仓机构个数（IHN）因子ICIR达0.84，反映机构抱团效应显著
。
 外资持股变动 
 定义 ：香港中央结算公司（北向资金）持股比例季度变化
。
 案例 ：香港中央结算公司在某股票中持股减少53.98%，可能预示外资撤离风险
。
四、风险预警类因子
 股东质押比例 
 定义 ：十大股东中质押股份占总股本比例
。
 阈值 ：质押比例>30%可能引发平仓风险（如房地产企业暴雷前预警）
。
 股权冻结/受限比例 
 计算 ：流通受限股份占总股本比例
。
 应用 ：受限比例突增（如从5%升至15%）可能反映股东资金链问题
。
五、股东类型多样性因子
 股东性质分布熵值 
 定义 ：计算个人、机构、外资等股东类型的分布离散度
。
 逻辑 ：熵值越高表明股东背景越多元，治理结构更平衡（如外资+机构+个人组合优于单一类型）
。
 实际控制人持股稳定性 
 指标 ：实控人连续3年持股比例波动率
。
 案例 ：实控人持股比例从57.77%降至40%可能预示控制权争夺风险
。
因子应用建议
 数据频率 ：十大股东数据多为季度更新，需与高频价量因子结合使用
。
 行业适配 ：
周期行业：侧重股权集中度（CR10）+股东质押比例；
科技行业：关注机构持股变动+研发投入股东占比
。
 复合因子构建 ：
光大证券的股东大类因子 （HN_z+LHRD+IHN）年化收益19%，夏普比率2.93
。
可叠加估值因子（如低PE+高CR10）提升策略稳定性
。
风险提示
 数据滞后性 ：季度数据可能错过短期事件冲击（如突发减持）
。
 行业异质性 ：股权集中度阈值需按行业调整（如金融业CR10普遍高于制造业）
。
 因子失效 ：需监控Rank IC衰减，如连续两季度IC<0.05时暂停使用
。
"""
###############################################################
class GovernanceMetrics(Metrics, KlineDetermination):
    """
    公司治理指标
    指标命名规则：指标名_计算窗口
    """

    @validate_literal_params
    def __init__(
            self,
            shareholders: pd.DataFrame,
            circulating_shareholders: pd.DataFrame,
            methods: list[str],
            function_map: dict[str, str]
    ):
        """
        :param shareholders: 股东数据
        :param circulating_shareholders: 流通股东数据
        :param methods: 需要实现的方法
        :param function_map: 已定义的方法对应方法名
        """
        self.shareholders_data = self._merger_data(shareholders, circulating_shareholders)
        self.metrics = pd.DataFrame()
        self.function_map = function_map
        self.methods = methods

    # --------------------------
    # 初始化数据处理 方法
    # --------------------------
    @staticmethod
    def _merger_data(
            shareholders: pd.DataFrame,
            circulating_shareholders: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        数据合并：十大股东 + 十大流通股东
        :param shareholders: 股东数据
        :param circulating_shareholders: 流通股东数据
        :return 合并后的数据
        """
        return pd.concat(
            [
                shareholders,
                circulating_shareholders.add_prefix("流通")
                .rename(
                    columns={f"流通占流通股本持股比例": "占流通股本持股比例"}
                )
            ],
            axis=1
        )

    # --------------------------
    # 公开 API 方法
    # --------------------------
    def calculate(
            self
    ) -> None:
        """计算接口"""
        for method in self.methods:
            if method in self.function_map:
                method_name = self.function_map.get(method)
                if method_name and hasattr(self, method_name):
                    getattr(self, method_name)()
                else:
                    raise ValueError(f"未实现的方法: {method}")
            else:
                raise ValueError(f"未定义的指标: {method}")

    # --------------------------
    # 量价 私有方法
    # --------------------------
    def _shareholding_cr10(self) -> None:
        """前十大股东合计占比"""
        self.metrics["CR10"] = self.shareholders_data.groupby(level=0)["占总股本持股比例"].sum()

    def _circulating_shareholding_cr10(self) -> None:
        """前十大流通股东合计占比"""
        self.metrics["流通CR10"] = self.shareholders_data.groupby(level=0)["占流通股本持股比例"].sum()

    def _major_shareholding(self) -> None:
        """大股东合计占比"""
        self.metrics["大股东合计占比"] = (
            self.shareholders_data.groupby(level=0)["占总股本持股比例"]
            .agg(lambda x: x[x>=5].sum())
        )

    def _shareholding_cr10_hhi(self) -> None:
        """
        ‌赫芬达尔指数 = 前10大股东持股比例的平方和
        """
        self.metrics["赫芬达尔指数"] = (
            self.shareholders_data.groupby(level=0)["占总股本持股比例"]
            .agg(lambda x: ((x/100)**2).sum())
        )

    def _circulating_shareholding_cr10_hhi(self) -> None:
        """
        ‌赫芬达尔指数 = 前10大流通股东持股比例的平方和
        """
        self.metrics["流通赫芬达尔指数"] = (
            self.shareholders_data.groupby(level=0)["占流通股本持股比例"]
            .agg(lambda x: ((x/100)**2).sum())
        )

    def _shareholders_z_index(self) -> None:
        """
        Z指数 = 第一大股东持股比例 / 第二大股东持股比例
        """
        self.metrics["Z指数"] = (
            self.shareholders_data.groupby(level=0).apply(
                lambda x: x[x["名次"] == 1]["占总股本持股比例"].values[0] /
                          x[x["名次"] == 2]["占总股本持股比例"].values[0]
            )
        )

    def _circulating_shareholders_z_index(self) -> None:
        """
        Z指数 = 第一大股东持股比例 / 第二大股东持股比例
        """
        self.metrics["流通Z指数"] = (
            self.shareholders_data.groupby(level=0).apply(
                lambda x: x[x["名次"] == 1]["占流通股本持股比例"].values[0] /
                          x[x["名次"] == 2]["占流通股本持股比例"].values[0]
            )
        )

    @depends_on("CR10")
    def _top_ten_shareholding_change_rate(self) -> None:
        """
        前十大股东持股变化率 = 前十大股东持股总额变化率
        """
        self.metrics["前十大股东持股变化率"] = self.metrics["CR10"].diff(1)

    @depends_on("大股东合计占比")
    def _major_shareholding_change_rate(self) -> None:
        """
        大股东持股变化率 = 大股东持股总额变化率
        """
        self.metrics["大股东持股变化率"] = self.metrics["大股东合计占比"].diff(1)

    def _institutional_shareholding(self) -> None:
        """
        机构持股比例
        流通股东性质：证券公司、私募基金、社保基金、保险产品、信托计划、集合理财计划、
                   全国社保基金、证券投资基金、基本养老基金、其他理财产品、QFII
        流通股东名称：中央汇金资产管理有限责任公司、中国证券金融股份有限公司
        """
        nature = [
            "证券公司", "私募基金", "社保基金", "保险产品", "信托计划", "集合理财计划",
            "全国社保基金", "证券投资基金", "基本养老基金", "其他理财产品", "QFII"
        ]
        name = [
            "中央汇金资产管理有限责任公司", "中国证券金融股份有限公司"
        ]
        filtered = self.shareholders_data[
            self.shareholders_data["流通股东性质"].isin(nature) |
            self.shareholders_data["流通股东名称"].isin(name)
        ]

        self.metrics["机构持股比例"] = filtered.groupby("date")["占流通股本持股比例"].sum()

    def _foreign_shareholding(self) -> None:
        """
        外资持股比例
        流通股东性质：QFII
        流通股东名称：香港中央结算有限公司
        """
        nature = ["QFII"]
        name = ["香港中央结算有限公司"]
        filtered = self.shareholders_data[
            self.shareholders_data["流通股东性质"].isin(nature) |
            self.shareholders_data["流通股东名称"].isin(name)
        ]

        self.metrics["外资持股比例"] = filtered.groupby("date")["占流通股本持股比例"].sum()
