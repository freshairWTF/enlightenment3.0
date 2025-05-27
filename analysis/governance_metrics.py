"""
公司治理指标
一、股权集中度类因子
二、股东行为动态因子
三、机构参与度因子
四、风险预警类因子
五、股东类型多样性因子
"""
import pandas as pd

from base_metrics import depends_on
from constant.type_ import validate_literal_params

"""
标记半衰期
根据半衰期筛选因子  得到 长、中、短期的选股集合

公司治理指标

行业指标
赫芬达尔指数：测算行业集中度
"""


###############################################################
class GovernanceMetrics:
    """
    公司治理指标
    """

    @validate_literal_params
    def __init__(
            self,
            shareholders: pd.DataFrame,
            # circulating_shareholders: pd.DataFrame,
            methods: list[str],
            function_map: dict[str, str]
    ):
        """
        :param shareholders: 股东数据
        # :param circulating_shareholders: 流通股东数据
        :param methods: 需要实现的方法
        :param function_map: 已定义的方法对应方法名
        """
        self.shareholders_data = shareholders
        # self.shareholders_data = self._merger_data(shareholders, circulating_shareholders)
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
    def _top_ten_shareholding(self) -> None:
        """前十大股东持股比例"""
        self.metrics["前十大股东持股比例"] = self.shareholders_data.groupby(level=0)["占总股本持股比例"].sum()

    @depends_on("前十大股东持股比例")
    def _top_ten_shareholding_change_rate(self) -> None:
        """前十大股东持股比例"""
        self.metrics["前十大股东持股变化率"] = self.metrics["前十大股东持股比例"].diff(1)

    def _circulating_top_ten_shareholding(self) -> None:
        """流通前十大股东持股比例"""
        self.metrics["流通前十大股东持股比例"] = self.shareholders_data.groupby(level=0)["占流通股本持股比例"].sum()

    @depends_on("流通前十大股东持股比例")
    def _circulating_top_ten_shareholding_change_rate(self) -> None:
        """流通前十大股东持股变化率"""
        self.metrics["流通前十大股东持股变化率"] = self.metrics["流通前十大股东持股比例"].diff(1)

    def _major_shareholding(self) -> None:
        """大股东持股比例"""
        self.metrics["大股东持股比例"] = (
            self.shareholders_data.groupby(level=0)["占总股本持股比例"]
            .agg(lambda x: x[x>=5].sum())
        )

    @depends_on("大股东持股比例")
    def _major_shareholding_change_rate(self) -> None:
        """大股东持股变化率"""
        self.metrics["大股东持股变化率"] = self.metrics["大股东持股比例"].diff(1)

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

    @depends_on("机构持股比例")
    def _institutional_shareholding_change_rate(self) -> None:
        """
        机构持股变化率 = 机构持股比例变化率
        """
        self.metrics["机构持股变化率"] = self.metrics["机构持股比例"].diff(1)

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

    @depends_on("外资持股比例")
    def _foreign_shareholding_change_rate(self) -> None:
        """
        外资持股变化率 = 外资持股比例变化率
        """
        self.metrics["外资持股变化率"] = self.metrics["外资持股比例"].diff(1)

    def _top_ten_shareholding_hhi(self) -> None:
        """
        ‌赫芬达尔指数 = 前10大股东持股比例的平方和
        """
        self.metrics["赫芬达尔指数"] = (
            self.shareholders_data.groupby(level=0)["占总股本持股比例"]
            .agg(lambda x: ((x/100)**2).sum())
        )

    def _circulating_top_ten_shareholding_hhi(self) -> None:
        """
        ‌赫芬达尔指数 = 前10大流通股东持股比例的平方和
        """
        self.metrics["流通赫芬达尔指数"] = (
            self.shareholders_data.groupby(level=0)["占流通股本持股比例"]
            .agg(lambda x: ((x/100)**2).sum())
        )

    def _z_index(self) -> None:
        """
        z指数 = 第一大股东持股比例 / 第二大股东持股比例
        """
        self.metrics["z指数"] = (
            self.shareholders_data.groupby(level=0).apply(
                lambda x: x[x["名次"] == 1]["占总股本持股比例"].values[0] /
                          x[x["名次"] == 2]["占总股本持股比例"].values[0]
            )
        )

    def _circulating_z_index(self) -> None:
        """
        流通z指数 = 第一大股东持股比例 / 第二大股东持股比例
        """
        self.metrics["流通z指数"] = (
            self.shareholders_data.groupby(level=0).apply(
                lambda x: x[x["名次"] == 1]["占流通股本持股比例"].values[0] /
                          x[x["名次"] == 2]["占流通股本持股比例"].values[0]
            )
        )
