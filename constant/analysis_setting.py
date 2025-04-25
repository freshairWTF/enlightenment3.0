from data_loader import DataLoader
from constant.path_config import DataPATH
from constant.draw_specs import BasicChartSpecs, QuadrantsChartSpecs


#############################################################
class Individual:

    def __init__(self):
        # 财务模块
        self.finance = self.Finance()
        # 估值模块
        self.valuation = self.Valuation()
        # 量价模块
        self.kline = self.Kline()
        # 统计模块
        self.stat = self.Statistics()
        # 可视化模块
        self.visualization = self.Visualization()

    class Finance:
        def __init__(self):
            """
            盈利能力：衡量企业赚取利润的能力。
            运营能力：反映企业资产运营效率和管理水平。
            偿债能力：评估企业偿还债务的能力。
            成长能力：衡量企业未来发展潜力和增长能力。
            现金流量：分析企业现金流入和流出的情况。
            资本结构：反映企业资金来源和构成。
            费用控制：评估企业对各项费用的控制能力。
            资产质量：衡量企业资产的安全性和价值。
            税务管理：反映企业税务负担和管理效率。
            其他：未明确归类的指标，通常为基础数据或辅助分析指标。
            """
            # 基础报表
            self.basic_reports = {
                "资产负债表": [
                    "非流动资产合计", "少数股东权益", "应收票据", "应收账款", "实收资本",
                    "库存股", "存货", "所有者权益", "负债和所有者权益"
                ],
                "利润表": [
                    "利润总额", "资产减值损失", "信用减值损失", "归属于母公司所有者的净利润",
                    "净利润", "营业利润", "营业收入"
                ],
                "现金流量表": [
                    "投资活动现金流入小计", "投资活动现金流出小计", "收回投资所收到的现金",
                    "取得投资收益所收到的现金", "处置固定资产、无形资产和其他长期资产所收回的现金净额",
                    "收到的其他与投资活动有关的现金", "购建固定资产、无形资产和其他长期资产支付的现金",
                    "投资所支付的现金", "取得子公司及其他营业单位支付的现金净额", "支付的其他与投资活动有关的现金",
                    "投资活动产生的现金流量净额", "筹资活动现金流入小计", "筹资活动现金流出小计",
                    "筹资活动产生的现金流量净额", "吸收投资收到的现金", "取得借款收到的现金",
                    "发行债券收到的现金", "收到其他与筹资活动有关的现金", "偿还债务支付的现金",
                    "分配股利、利润或偿付利息所支付的现金", "支付其他与筹资活动有关的现金", "经营活动现金流入小计",
                    "经营活动现金流出小计", "经营活动产生的现金流量净额", "销售商品、提供劳务收到的现金",
                    "收到的税费返还", "收到的其他与经营活动有关的现金", "购买商品、接受劳务支付的现金",
                    "支付给职工以及为职工支付的现金", "支付的各项税费", "支付的其他与经营活动有关的现金",
                ]
            }
            # 财务分析
            self.financial_analysis = {
                "盈利能力": [
                    "毛利率", "核心利润率", "营业净利率", "核心利润净利率", "权益净利率",
                    "归母权益净利率", "总资产净利率", "净经营资产净利率", "经营净利率",
                    "销售利润率", "经营差异率", "杠杆贡献率", "核心利润占比", "经营性资产收益率",
                    "金融性资产收益率", "毛利", "投资利润", "杂项利润", "营业外收支净额",
                    "息税前利润", "息税前利润率"
                ],
                "运营能力": [
                    "存货周转天数", "应收票据及应收账款周转天数", "应付票据及应付账款周转天数",
                    "现金转换周期", "固定资产周转率", "总资产周转率", "单位营收所需的经营性营运资本",
                    "营业周期", "固定资产周转天数", "流动资产周转天数", "非流动资产周转天数",
                    "总资产周转天数", "净经营资产周转率", "固定资产原值推动力", "固定资产净值推动力",
                ],
                "偿债能力": [
                    "流动比率", "速动比率", "现金比率", "资产负债率", "有息负债率",
                    "债务资本化比率", "长期资本化比率", "利息保障倍数", "现金流量利息保障倍数",
                    "净财务杠杆", "现金流量比率"
                ],
                "成长能力": [
                    "内含增长率", "可持续增长率", "利润留存率", "股利支付率", "扩张倍数",
                    "收缩倍数"
                ],
                "现金流量": [
                    "经营现金流量", "债务现金流量", "股权现金流量", "核心利润获现率",
                    "收现比", "净现比", "企业自由现金流", "股权自由现金流",
                    "无杠杆自由现金流", "杠杆自由现金流",
                ],
                "资本结构": [
                    "金融性负债占比", "经营性负债占比", "股东入资占比", "利润留存占比",
                    "流动资产占比", "货币资金占比", "经营性资产占比", "金融性资产占比",
                    "应付款比率", "预收款比率", "上下游资金占用"
                ],
                "费用控制": [
                    "销售费用率", "管理费用率", "财务费用率", "研发费用率", "期间费用率",
                    "毛销差", "营业税金负担率", "期间费用"
                ],
                "资产质量": [
                    "信用减值损失净资产占比", "资产减值损失净资产占比", "减值损失核心利润占比",
                    "减值损失"
                ],
                "税务管理": [
                    "实际税率"
                ],
                "其他": [
                    "经营性流动资产", "经营性流动负债", "经营性营运资本", "经营性长期资产",
                    "经营性长期负债", "净经营性长期资产", "净经营资产", "金融性负债",
                    "金融性资产", "净负债", "经营净利润", "金融净利润", "beneish_m_score"
                ]
            }

    class Valuation:
        def __init__(self):
            # 基础指标
            self.basic_metrics = [
                "市值", "对数市值", "市销率", "市净率", "核心利润市盈率", "核心利润盈利市值比",
            ]
            # 衍生指标
            self.derived_metrics = [
                "周期市盈率", "市盈增长率", "股息率", "核心利润实际收益率", "企业自由现金流现值"
            ]

    class Kline:
        def __init__(self):
            self.kline = {
                "市场贝塔": [1]
            }

    class Statistics:
        def __init__(self):
            self.stats = ["同比", "环比", "复合增长率", "归一化"]

    class Visualization:
        def __init__(self):
            parameters = DataLoader.load_yaml_file(DataPATH.FINANCIAL_METRICS, "PARAMETERS_OF_EASTMONEY")
            # page名
            self.pages_name = [
                "管理用财务报表", "滚动财务", "财务", "估值", "其他"
            ]
            # 图表配置
            self.pages_config = {
                "管理用财务报表": {
                    "_upper_lower_dichotomy-1": QuadrantsChartSpecs(
                        title="总资产",
                        ul_data_source="rolling_financial",
                        ul_column="负债和所有者权益",
                        ll_data_source="rolling_financial",
                        ll_column="负债和所有者权益_yoy"
                    ),
                    "_blpp-1": QuadrantsChartSpecs(
                        title="资产来源",
                        ul_data_source="rolling_financial",
                        ul_column=["金融性负债", "经营性负债", "股东入资", "利润留存"],
                        ll_data_source="rolling_financial",
                        ll_column=["金融性负债_yoy", "经营性负债_yoy", "股东入资_yoy", "利润留存_yoy"],
                        ll_chart="line",
                        ur_column=parameters["股东入资"],
                        ur_data_source="rolling_financial",
                        lr_column=parameters["利润留存"],
                        lr_data_source="rolling_financial",
                        main_pie_show=True
                    ),
                    "_blpp-2": QuadrantsChartSpecs(
                        title="资产流向",
                        ul_data_source="rolling_financial",
                        ul_column=["经营性资产", "金融性资产"],
                        ll_data_source="rolling_financial",
                        ll_column=["经营性资产_yoy", "金融性资产_yoy"],
                        ll_chart="line",
                        ur_column=parameters["经营性资产"],
                        ur_data_source="rolling_financial",
                        lr_column=parameters["金融性资产"],
                        lr_data_source="rolling_financial",
                        main_pie_show=True
                    ),
                    "_blpp-3": QuadrantsChartSpecs(
                        title="资产流动性",
                        ul_data_source="rolling_financial",
                        ul_column=["流动资产合计", "非流动资产合计"],
                        ll_data_source="rolling_financial",
                        ll_column=["流动资产合计_yoy", "非流动资产合计_yoy"],
                        ll_chart="line",
                        ur_column=parameters["流动资产"],
                        ur_data_source="rolling_financial",
                        lr_column=parameters["非流动资产"],
                        lr_data_source="rolling_financial",
                        main_pie_show=True
                    ),
                    "_blpp-4": QuadrantsChartSpecs(
                        title="经营性流动资产",
                        ul_data_source="rolling_financial",
                        ul_column=["经营性流动资产", "经营性流动负债"],
                        ll_data_source="rolling_financial",
                        ll_column=["经营性营运资本", "经营性营运资本_yoy"],
                        ll_chart="line",
                        ur_column=parameters["经营性流动资产"],
                        ur_data_source="rolling_financial",
                        lr_column=parameters["经营性流动负债"],
                        lr_data_source="rolling_financial",
                        main_pie_show=True
                    ),
                    "_blpp-5": QuadrantsChartSpecs(
                        title="经营性长期资产",
                        ul_data_source="rolling_financial",
                        ul_column=["经营性长期资产", "经营性长期负债"],
                        ll_data_source="rolling_financial",
                        ll_column=["净经营性长期资产", "净经营性长期资产_yoy"],
                        ll_chart="line",
                        ur_column=parameters["经营性长期资产"],
                        ur_data_source="rolling_financial",
                        lr_column=parameters["经营性长期负债"],
                        lr_data_source="rolling_financial",
                        main_pie_show=True
                    ),
                    "_blpp-6": QuadrantsChartSpecs(
                        title="金融性资产",
                        ul_data_source="rolling_financial",
                        ul_column=["金融性资产", "金融性负债"],
                        ll_data_source="rolling_financial",
                        ll_column=["净负债", "净负债_yoy"],
                        ll_chart="line",
                        ur_column=parameters["金融性资产"],
                        ur_data_source="rolling_financial",
                        lr_column=parameters["金融性负债"],
                        lr_data_source="rolling_financial",
                        main_pie_show=True
                    ),
                    "_quadrants-1": QuadrantsChartSpecs(
                        title="净经营资产",
                        ul_data_source="rolling_financial",
                        ul_column="净经营资产",
                        ll_data_source="rolling_financial",
                        ll_column="净经营资产_yoy",
                        ll_chart="line",
                        ur_column=["净负债", "所有者权益"],
                        ur_data_source="rolling_financial",
                        lr_column=["净负债_yoy", "所有者权益_yoy"],
                        lr_data_source="rolling_financial",
                        lr_chart="line"
                    ),
                    "_quadrants-2": QuadrantsChartSpecs(
                        title="利润来源",
                        ul_data_source="rolling_financial",
                        ul_column="经营净利润",
                        ll_data_source="rolling_financial",
                        ll_column="经营净利润_yoy",
                        ll_chart="line",
                        ur_data_source="rolling_financial",
                        ur_column="金融净利润",
                        lr_data_source="rolling_financial",
                        lr_column="金融净利润_yoy",
                        lr_chart="line"
                    ),
                    "_upper_lower_dichotomy-2": QuadrantsChartSpecs(
                        title="权益净利率",
                        ul_data_source="rolling_financial",
                        ul_column="权益净利率",
                        ll_data_source="rolling_financial",
                        ll_column="归母权益净利率"
                    ),
                    "_quadrants-4": QuadrantsChartSpecs(
                        title="权益净利率拆解"
                              "\n权益净利率=净经营资产净利率+杠杆贡献率"
                              "\n杠杆贡献率=经营差异率*净财务杠杆"
                              "\n经营差异率=净经营资产净利率-税后利息率",
                        ul_data_source="rolling_financial",
                        ul_column="权益净利率",
                        ll_data_source="rolling_financial",
                        ll_column=["净经营资产净利率", "杠杆贡献率"],
                        ur_data_source="rolling_financial",
                        ur_column=["经营差异率", "净财务杠杆"],
                        lr_data_source="rolling_financial",
                        lr_column=["净经营资产净利率", "税后利息率"],
                    ),
                    "_quadrants-5": QuadrantsChartSpecs(
                        title="净经营资产净利率拆解"
                              "\n净经营资产净利率=经营净利率*净经营资产周转率"
                              "\n经营净利率=经营净利润/营业收入"
                              "\n净经营资产周转率=营业收入/净经营资产",
                        ul_data_source="rolling_financial",
                        ul_column="净经营资产净利率",
                        ll_data_source="rolling_financial",
                        ll_column=["经营净利率", "净经营资产周转率"],
                        ur_data_source="rolling_financial",
                        ur_column=["经营净利润", "营业收入"],
                        lr_data_source="rolling_financial",
                        lr_column=["营业收入", "净经营资产"],
                    ),
                },
                "滚动财务": {
                    "_quadrants-1": QuadrantsChartSpecs(
                        title="偿债能力",
                        ul_data_source="rolling_financial",
                        ul_column=["流动比率", "速动比率", "现金比率", "现金流量比率"],
                        ll_data_source="rolling_financial",
                        ll_column=["有息负债率", "资产负债率", "长期资本化比率"],
                        ur_data_source="rolling_financial",
                        ur_column=["利息保障倍数", "现金流量利息保障倍数"],
                    ),
                    "_quadrants-2": QuadrantsChartSpecs(
                        title="盈利能力",
                        ul_data_source="rolling_financial",
                        ul_column=["营业收入", "毛利", "核心利润"],
                        ll_data_source="rolling_financial",
                        ll_column=["营业收入_yoy", "毛利_yoy", "核心利润_yoy"],
                        ll_chart="line",
                        ur_data_source="rolling_financial",
                        ur_column=["利润总额", "净利润", "归属于母公司所有者的净利润"],
                        lr_data_source="rolling_financial",
                        lr_column=["利润总额_yoy", "净利润_yoy", "归属于母公司所有者的净利润_yoy"],
                        lr_chart="line"
                    ),
                    "_quadrants-3": QuadrantsChartSpecs(
                        title="盈利能力",
                        ul_data_source="rolling_financial",
                        ul_column=["毛利率", "毛销差"],
                        ll_data_source="rolling_financial",
                        ll_column="核心利润率",
                        ur_data_source="rolling_financial",
                        ur_column=["销售利润率", "营业净利率"],
                        lr_data_source="rolling_financial",
                        lr_column=["营业税金负担率"]
                    ),
                    "_blpp-1": QuadrantsChartSpecs(
                        title="利润来源",
                        ul_data_source="rolling_financial",
                        ul_column=["核心利润", "投资利润", "杂项利润", "营业外收支净额"],
                        ll_data_source="rolling_financial",
                        ll_column=["核心利润_yoy", "投资利润_yoy", "杂项利润_yoy", "营业外收支净额_yoy"],
                        ll_chart="line",
                        main_pie_show=True
                    ),
                    "_quadrants-4": QuadrantsChartSpecs(
                        title="费用",
                        ul_data_source="rolling_financial",
                        ul_column=["销售费用", "管理费用", "财务费用", "研发费用", "期间费用"],
                        ll_data_source="rolling_financial",
                        ll_column=["销售费用_yoy", "管理费用_yoy", "财务费用_yoy", "研发费用_yoy", "期间费用_yoy"],
                        ll_chart="line",
                        ur_data_source="rolling_financial",
                        ur_column=["销售费用率", "管理费用率", "财务费用率", "研发费用率", "期间费用率"],
                        ur_chart="line",
                        lr_data_source="rolling_financial",
                        lr_column=["实际税率"],
                        lr_chart="line"
                    ),
                    "_quadrants-5": QuadrantsChartSpecs(
                        title="减值",
                        ul_data_source="rolling_financial",
                        ul_column=["减值损失", "资产减值损失", "信用减值损失"],
                        ll_data_source="rolling_financial",
                        ll_column=["减值损失_yoy", "资产减值损失_yoy", "信用减值损失_yoy"],
                        ll_chart="line",
                        ur_data_source="rolling_financial",
                        ur_column="减值损失核心利润占比",
                        lr_data_source="rolling_financial",
                        lr_column=["资产减值损失净资产占比", "信用减值损失净资产占比"]
                    ),
                    "_quadrants-6": QuadrantsChartSpecs(
                        title="非核心利润来源",
                        ul_data_source="rolling_financial",
                        ul_column=["公允价值变动收益", "投资收益"],
                        ll_data_source="rolling_financial",
                        ll_column=["其他收益", "资产处置收益"],
                        ur_data_source="rolling_financial",
                        ur_column=["营业外收入", "营业外支出"]
                    ),
                    "_quadrants-7": QuadrantsChartSpecs(
                        title="利润质量",
                        ul_data_source="rolling_financial",
                        ul_column=["核心利润获现率", "收现比", "净现比"],
                        ll_data_source="rolling_financial",
                        ll_column=["核心利润获现率_yoy", "收现比_yoy", "净现比_yoy"],
                        ll_chart="line",
                        ur_data_source="rolling_financial",
                        ur_column="经营活动产生的现金流量净额"
                    ),
                    "_quadrants-8": QuadrantsChartSpecs(
                        title="利润质量",
                        ul_data_source="rolling_financial",
                        ul_column="上下游资金占用",
                        ll_data_source="rolling_financial",
                        ll_column=["应付款比率", "预收款比率"],
                        ur_data_source="rolling_financial",
                        ur_column="单位营收所需的经营性营运资本",
                        lr_data_source="rolling_financial",
                        lr_column=["经营性营运资本", "营业收入"]
                    ),
                    "_quadrants-9": QuadrantsChartSpecs(
                        title="运营效率",
                        ul_data_source="rolling_financial",
                        ul_column=["营业周期", "现金转换周期"],
                        ll_data_source="rolling_financial",
                        ll_column=["存货周转天数", "应收票据及应收账款周转天数", "应付票据及应付账款周转天数"],
                        ur_data_source="rolling_financial",
                        ur_column=["总资产周转天数", "流动资产周转天数", "非流动资产周转天数", "固定资产周转天数"],
                        lr_data_source="rolling_financial",
                        lr_column=["总资产周转天数_yoy", "流动资产周转天数_yoy", "非流动资产周转天数_yoy",
                                   "固定资产周转天数_yoy"],
                        lr_chart="line"
                    ),
                    "_quadrants-10": QuadrantsChartSpecs(
                        title="成长潜力",
                        ul_data_source="rolling_financial",
                        ul_column=["固定资产原值推动力", "固定资产净值推动力"],
                        ll_data_source="rolling_financial",
                        ll_column=["营业收入", "固定资产净额", "固定资产原值"],
                        ur_data_source="rolling_financial",
                        ur_column=["扩张倍数", "收缩倍数"]
                    ),
                    "_quadrants-11": QuadrantsChartSpecs(
                        title="成长潜力",
                        ul_data_source="rolling_financial",
                        ul_column="内含增长率",
                        ll_data_source="rolling_financial",
                        ll_column=["营业净利率", "净经营资产周转率", "利润留存率"],
                        ur_data_source="rolling_financial",
                        ur_column="可持续增长率",
                        lr_data_source="rolling_financial",
                        lr_column=["营业净利率", "净经营资产周转率", "净经营资产权益乘数", "利润留存率"],
                    ),
                    "_quadrants-12": QuadrantsChartSpecs(
                        title="现金流",
                        ul_data_source="rolling_financial",
                        ul_column="企业自由现金流",
                        ll_data_source="rolling_financial",
                        ll_column="股权自由现金流",
                        ur_data_source="rolling_financial",
                        ur_column="无杠杆自由现金流",
                        lr_data_source="rolling_financial",
                        lr_column="杠杆自由现金流",
                    )
                },
                "财务": {
                    "_quadrants-1": QuadrantsChartSpecs(
                        title="偿债能力",
                        ul_data_source="financial",
                        ul_column=["流动比率", "速动比率", "现金比率", "现金流量比率"],
                        ll_data_source="financial",
                        ll_column=["有息负债率", "资产负债率", "长期资本化比率"],
                        ur_data_source="financial",
                        ur_column=["利息保障倍数", "现金流量利息保障倍数"],
                    ),
                    "_quadrants-2": QuadrantsChartSpecs(
                        title="盈利能力",
                        ul_data_source="financial",
                        ul_column=["营业收入", "毛利", "核心利润"],
                        ll_data_source="financial",
                        ll_column=["营业收入_yoy", "毛利_yoy", "核心利润_yoy"],
                        ll_chart="line",
                        ur_data_source="financial",
                        ur_column=["利润总额", "净利润", "归属于母公司所有者的净利润"],
                        lr_data_source="financial",
                        lr_column=["利润总额_yoy", "净利润_yoy", "归属于母公司所有者的净利润_yoy"],
                        lr_chart="line"
                    ),
                    "_quadrants-3": QuadrantsChartSpecs(
                        title="盈利能力",
                        ul_data_source="financial",
                        ul_column=["毛利率", "毛销差"],
                        ll_data_source="financial",
                        ll_column="核心利润率",
                        ur_data_source="financial",
                        ur_column=["销售利润率", "营业净利率"],
                        lr_data_source="financial",
                        lr_column=["营业税金负担率"]
                    ),
                    "_blpp-1": QuadrantsChartSpecs(
                        title="利润来源",
                        ul_data_source="financial",
                        ul_column=["核心利润", "投资利润", "杂项利润", "营业外收支净额"],
                        ll_data_source="financial",
                        ll_column=["核心利润_yoy", "投资利润_yoy", "杂项利润_yoy", "营业外收支净额_yoy"],
                        ll_chart="line",
                        main_pie_show=True
                    ),
                    "_quadrants-4": QuadrantsChartSpecs(
                        title="费用",
                        ul_data_source="financial",
                        ul_column=["销售费用", "管理费用", "财务费用", "研发费用", "期间费用"],
                        ll_data_source="financial",
                        ll_column=["销售费用_yoy", "管理费用_yoy", "财务费用_yoy", "研发费用_yoy", "期间费用_yoy"],
                        ll_chart="line",
                        ur_data_source="financial",
                        ur_column=["销售费用率", "管理费用率", "财务费用率", "研发费用率", "期间费用率"],
                        ur_chart="line",
                        lr_data_source="financial",
                        lr_column=["实际税率"],
                        lr_chart="line"
                    ),
                    "_quadrants-5": QuadrantsChartSpecs(
                        title="减值",
                        ul_data_source="financial",
                        ul_column=["减值损失", "资产减值损失", "信用减值损失"],
                        ll_data_source="financial",
                        ll_column=["减值损失_yoy", "资产减值损失_yoy", "信用减值损失_yoy"],
                        ll_chart="line",
                        ur_data_source="financial",
                        ur_column="减值损失核心利润占比",
                        lr_data_source="financial",
                        lr_column=["资产减值损失净资产占比", "信用减值损失净资产占比"]
                    ),
                    "_quadrants-6": QuadrantsChartSpecs(
                        title="非核心利润来源",
                        ul_data_source="financial",
                        ul_column=["公允价值变动收益", "投资收益"],
                        ll_data_source="financial",
                        ll_column=["其他收益", "资产处置收益"],
                        ur_data_source="financial",
                        ur_column=["营业外收入", "营业外支出"]
                    ),
                    "_quadrants-7": QuadrantsChartSpecs(
                        title="利润质量",
                        ul_data_source="financial",
                        ul_column=["核心利润获现率", "收现比", "净现比"],
                        ll_data_source="financial",
                        ll_column=["核心利润获现率_yoy", "收现比_yoy", "净现比_yoy"],
                        ll_chart="line",
                        ur_data_source="financial",
                        ur_column="经营活动产生的现金流量净额"
                    ),
                    "_quadrants-8": QuadrantsChartSpecs(
                        title="利润质量",
                        ul_data_source="financial",
                        ul_column="上下游资金占用",
                        ll_data_source="financial",
                        ll_column=["应付款比率", "预收款比率"],
                        ur_data_source="financial",
                        ur_column="单位营收所需的经营性营运资本",
                        lr_data_source="financial",
                        lr_column=["经营性营运资本", "营业收入"]
                    ),
                    "_quadrants-9": QuadrantsChartSpecs(
                        title="运营效率",
                        ul_data_source="financial",
                        ul_column=["营业周期", "现金转换周期"],
                        ll_data_source="financial",
                        ll_column=["存货周转天数", "应收票据及应收账款周转天数", "应付票据及应付账款周转天数"],
                        ur_data_source="financial",
                        ur_column=["总资产周转天数", "流动资产周转天数", "非流动资产周转天数", "固定资产周转天数"],
                        lr_data_source="financial",
                        lr_column=["总资产周转天数_yoy", "流动资产周转天数_yoy", "非流动资产周转天数_yoy", "固定资产周转天数_yoy"],
                        lr_chart="line"
                    ),
                    "_quadrants-10": QuadrantsChartSpecs(
                        title="成长潜力",
                        ul_data_source="financial",
                        ul_column=["固定资产原值推动力", "固定资产净值推动力",],
                        ll_data_source="financial",
                        ll_column=["营业收入", "固定资产净额", "固定资产原值"],
                        ur_data_source="financial",
                        ur_column=["扩张倍数", "收缩倍数"]
                    ),
                    "_quadrants-11": QuadrantsChartSpecs(
                        title="成长潜力",
                        ul_data_source="financial",
                        ul_column="内含增长率",
                        ll_data_source="financial",
                        ll_column=["营业净利率", "净经营资产周转率", "利润留存率"],
                        ur_data_source="financial",
                        ur_column="可持续增长率",
                        lr_data_source="financial",
                        lr_column=["营业净利率", "净经营资产周转率", "净经营资产权益乘数", "利润留存率"],
                    ),
                    "_quadrants-12": QuadrantsChartSpecs(
                        title="现金流",
                        ul_data_source="financial",
                        ul_column="企业自由现金流",
                        ll_data_source="financial",
                        ll_column="股权自由现金流",
                        ur_data_source="financial",
                        ur_column="无杠杆自由现金流",
                        lr_data_source="financial",
                        lr_column="杠杆自由现金流",
                    )
                },
                "估值": {
                    "_upper_lower_dichotomy-1": QuadrantsChartSpecs(
                        title="市值",
                        ul_column="市值",
                        ul_data_source="valuation",
                        ll_column="市值",
                        ll_data_source="valuation_normalized",
                        ll_chart="line"
                    ),
                    "_upper_lower_dichotomy-2": QuadrantsChartSpecs(
                        title="估值指标",
                        ul_column=["核心利润市盈率", "市净率", "市销率"],
                        ul_data_source="valuation",
                        ll_column=["核心利润市盈率", "市净率", "市销率"],
                        ll_data_source="valuation_normalized",
                        ll_chart="line"
                    ),
                    "_upper_lower_dichotomy-3": QuadrantsChartSpecs(
                        title="周期估值指标",
                        ul_column="周期市盈率",
                        ul_data_source="valuation",
                        ll_column="周期市盈率",
                        ll_data_source="valuation_normalized",
                        ll_chart="line"
                    ),
                    "_upper_lower_dichotomy-4": QuadrantsChartSpecs(
                        title="成长估值指标",
                        ul_column="市盈增长率",
                        ul_data_source="valuation",
                        ll_column="市盈增长率",
                        ll_data_source="valuation_normalized",
                        ll_chart="line"
                    ),
                    "_upper_lower_dichotomy-5": QuadrantsChartSpecs(
                        title="保底估值指标",
                        ul_column="股息率",
                        ul_data_source="valuation",
                        ll_column="股息率",
                        ll_data_source="valuation_normalized",
                        ll_chart="line"
                    ),
                    "_upper_lower_dichotomy-6": QuadrantsChartSpecs(
                        title="保底估值指标",
                        ul_column="核心利润实际收益率",
                        ul_data_source="valuation",
                        ll_column="核心利润实际收益率",
                        ll_data_source="valuation_normalized",
                        ll_chart="line"
                    ),
                    "_basic_line-1": BasicChartSpecs(
                        title="fcff折现",
                        data_source="valuation",
                        column="企业自由现金流折现值",
                    ),
                },
                "其他": {
                    "_basic_line-1": BasicChartSpecs(
                        title="Beneish M_Score",
                        data_source="rolling_financial",
                        column="beneish_m_score"
                    ),

                }
            }


#############################################################
class Normal:
    def __init__(self):
        # 财务模块
        self.finance = self.Finance()
        # 估值模块
        self.valuation = self.Valuation()
        # 量价模块
        self.kline = self.Kline()
        # 统计模块
        self.stat = self.Statistics()
        # 可视化模块
        self.visualization = self.Visualization()

    class Finance:
        def __init__(self):
            """
            盈利能力：衡量企业赚取利润的能力。
            运营能力：反映企业资产运营效率和管理水平。
            偿债能力：评估企业偿还债务的能力。
            成长能力：衡量企业未来发展潜力和增长能力。
            现金流量：分析企业现金流入和流出的情况。
            资本结构：反映企业资金来源和构成。
            费用控制：评估企业对各项费用的控制能力。
            资产质量：衡量企业资产的安全性和价值。
            税务管理：反映企业税务负担和管理效率。
            其他：未明确归类的指标，通常为基础数据或辅助分析指标。
            """
            # 基础报表
            self.basic_reports = {
                "资产负债表": ["负债和所有者权益", "所有者权益", "存货", "固定资产净额", "固定资产原值"],
                "利润表": ["净利润", "营业收入"],
                "现金流量表": ["经营活动产生的现金流量净额"]
            }
            # 财务分析
            self.financial_analysis = {
                "盈利能力": [
                    "毛利率", "核心利润率", "营业净利率", "核心利润净利率", "权益净利率",
                    "归母权益净利率", "总资产净利率", "净经营资产净利率", "经营净利率",
                    "销售利润率", "经营差异率", "杠杆贡献率", "核心利润占比", "经营性资产收益率",
                    "金融性资产收益率", "毛利", "核心利润"
                ],
                "运营能力": [
                    "存货周转天数", "应收票据及应收账款周转天数", "应付票据及应付账款周转天数",
                    "现金转换周期", "固定资产周转率", "总资产周转率", "单位营收所需的经营性营运资本",
                    "营业周期", "固定资产周转天数", "流动资产周转天数", "非流动资产周转天数",
                    "总资产周转天数", "净经营资产周转率", "固定资产原值推动力", "固定资产净值推动力",
                ],
                "偿债能力": [
                    "流动比率", "速动比率", "现金比率", "资产负债率", "有息负债率",
                    "债务资本化比率", "长期资本化比率", "利息保障倍数", "现金流量利息保障倍数",
                    "净财务杠杆", "现金流量比率"
                ],
                "成长能力": [
                    "内含增长率", "可持续增长率", "利润留存率", "股利支付率", "扩张倍数",
                    "收缩倍数"
                ],
                "现金流量": [
                    "经营现金流量", "债务现金流量", "股权现金流量", "核心利润获现率",
                    "收现比", "净现比", "企业自由现金流", "股权自由现金流",
                    "无杠杆自由现金流", "杠杆自由现金流",
                ],
                "资本结构": [
                    "金融性负债占比", "经营性负债占比", "股东入资占比", "利润留存占比",
                    "流动资产占比", "货币资金占比", "经营性资产占比", "金融性资产占比",
                    "应付款比率", "预收款比率", "上下游资金占用",
                ],
                "费用控制": [
                    "销售费用率", "管理费用率", "财务费用率", "研发费用率", "期间费用率",
                    "毛销差", "营业税金负担率"
                ],
                "资产质量": [
                    "信用减值损失净资产占比", "资产减值损失净资产占比", "减值损失核心利润占比"
                ],
                "税务管理": [
                    "税后利息率",
                ],
                "其他": [
                    "经营性流动资产", "经营性流动负债", "经营性营运资本", "经营性长期资产",
                    "经营性长期负债", "净经营性长期资产", "净经营资产", "金融性负债",
                    "金融性资产", "净负债", "经营净利润", "金融净利润"
                ]
            }

    class Valuation:
        def __init__(self):
            # 基础指标
            self.basic_metrics = [
                "市值", "对数市值", "市销率", "市净率", "核心利润市盈率", "核心利润盈利市值比"
            ]
            # 衍生指标
            self.derived_metrics = [
                "周期市盈率", "市盈增长率", "股息率", "核心利润实际收益率"
            ]

    class Kline:
        def __init__(self):
            self.kline = {}

    class Statistics:
        def __init__(self):
            self.stats = ["同比", "环比", "复合增长率", "归一化", "滚动归一化"]

    class Visualization:
        def __init__(self):
            # page名
            self.pages_name = [
                "权益净利率", "偿债能力", "盈利能力", "营运能力", "成长能力", "现金流", "其他指标", "估值"
            ]
            # 图表配置
            self.pages_config = {
                "权益净利率": {
                    "_basic_bar-1": BasicChartSpecs(
                        title="归母权益净利率",
                        data_source="rolling_financial",
                        column="归母权益净利率"
                    ),
                    "_basic_bar-2": BasicChartSpecs(
                        title="权益净利率",
                        data_source="rolling_financial",
                        column="权益净利率"
                    ),
                    "_basic_bar-3": BasicChartSpecs(
                        title="净经营资产净利率",
                        data_source="rolling_financial",
                        column="净经营资产净利率"
                    ),
                    "_basic_bar-4": BasicChartSpecs(
                        title="经营净利率",
                        data_source="rolling_financial",
                        column="经营净利率"
                    ),
                    "_basic_bar-5": BasicChartSpecs(
                        title="净经营资产周转率",
                        data_source="rolling_financial",
                        column="净经营资产周转率"
                    ),
                    "_basic_bar-6": BasicChartSpecs(
                        title="杠杆贡献率",
                        data_source="rolling_financial",
                        column="杠杆贡献率"
                    ),
                    "_basic_bar-7": BasicChartSpecs(
                        title="净财务杠杆",
                        data_source="rolling_financial",
                        column="净财务杠杆"
                    ),
                    "_basic_bar-8": BasicChartSpecs(
                        title="经营差异率",
                        data_source="rolling_financial",
                        column="经营差异率"
                    ),
                    "_basic_bar-9": BasicChartSpecs(
                        title="税后利息率",
                        data_source="rolling_financial",
                        column="税后利息率"
                    )
                },
                "偿债能力": {
                    "_upper_lower_dichotomy-1": QuadrantsChartSpecs(
                        title="资产规模",
                        ul_column="负债和所有者权益",
                        ul_data_source="rolling_financial",
                        ll_column="负债和所有者权益_yoy",
                        ll_data_source="rolling_financial"
                    ),
                    "_basic_bar-1": BasicChartSpecs(
                        title="流动比率",
                        data_source="rolling_financial",
                        column="流动比率"
                    ),
                    "_basic_bar-2": BasicChartSpecs(
                        title="速动比率",
                        data_source="rolling_financial",
                        column="速动比率"
                    ),
                    "_basic_bar-3": BasicChartSpecs(
                        title="现金比率",
                        data_source="rolling_financial",
                        column="现金比率"
                    ),
                    "_basic_bar-4": BasicChartSpecs(
                        title="现金流量比率",
                        data_source="rolling_financial",
                        column="现金流量比率"
                    ),
                    "_basic_bar-5": BasicChartSpecs(
                        title="资产负债率",
                        data_source="rolling_financial",
                        column="资产负债率"
                    ),
                    "_basic_bar-6": BasicChartSpecs(
                        title="有息负债率",
                        data_source="rolling_financial",
                        column="有息负债率"
                    ),
                    "_basic_bar-7": BasicChartSpecs(
                        title="长期资本化比率",
                        data_source="rolling_financial",
                        column="长期资本化比率"
                    ),
                    "_basic_bar-8": BasicChartSpecs(
                        title="利息保障倍数",
                        data_source="rolling_financial",
                        column="利息保障倍数"
                    ),
                    "_basic_bar-9": BasicChartSpecs(
                        title="现金流量利息保障倍数",
                        data_source="rolling_financial",
                        column="现金流量利息保障倍数"
                    )
                },
                "盈利能力": {
                    "_basic_bar-1": BasicChartSpecs(
                        title="营业收入",
                        data_source="rolling_financial",
                        column="营业收入"
                    ),
                    "_basic_bar-2": BasicChartSpecs(
                        title="营收百分位数",
                        data_source="rolling_financial",
                        column="营业收入_normalized"
                    ),
                    "_basic_bar-3": BasicChartSpecs(
                        title="营收增速",
                        data_source="rolling_financial",
                        column="营业收入_yoy"
                    ),
                    "_basic_bar-4": BasicChartSpecs(
                        title="毛利",
                        data_source="rolling_financial",
                        column="毛利"
                    ),
                    "_basic_bar-5": BasicChartSpecs(
                        title="毛利百分位数",
                        data_source="rolling_financial",
                        column="毛利_normalized"
                    ),
                    "_basic_bar-6": BasicChartSpecs(
                        title="毛利增速",
                        data_source="rolling_financial",
                        column="毛利_yoy"
                    ),
                    "_upper_lower_dichotomy-1": QuadrantsChartSpecs(
                        title="毛利率",
                        ul_column="毛利率",
                        ul_data_source="rolling_financial",
                        ll_column="毛利率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-2": QuadrantsChartSpecs(
                        title="销售费用率",
                        ul_column="销售费用率",
                        ul_data_source="rolling_financial",
                        ll_column="销售费用率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-3": QuadrantsChartSpecs(
                        title="管理费用率",
                        ul_column="管理费用率",
                        ul_data_source="rolling_financial",
                        ll_column="管理费用率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-4": QuadrantsChartSpecs(
                        title="财务费用率",
                        ul_column="财务费用率",
                        ul_data_source="rolling_financial",
                        ll_column="财务费用率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-5": QuadrantsChartSpecs(
                        title="研发费用率",
                        ul_column="研发费用率",
                        ul_data_source="rolling_financial",
                        ll_column="研发费用率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-6": QuadrantsChartSpecs(
                        title="期间费用率",
                        ul_column="期间费用率",
                        ul_data_source="rolling_financial",
                        ll_column="期间费用率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-14": QuadrantsChartSpecs(
                        title="信用减值损失净资产占比",
                        ul_column="信用减值损失净资产占比",
                        ul_data_source="rolling_financial",
                        ll_column="信用减值损失净资产占比_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-15": QuadrantsChartSpecs(
                        title="资产减值损失净资产占比",
                        ul_column="资产减值损失净资产占比",
                        ul_data_source="rolling_financial",
                        ll_column="资产减值损失净资产占比_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_basic_bar-7": BasicChartSpecs(
                        title="核心利润",
                        data_source="rolling_financial",
                        column="核心利润"
                    ),
                    "_basic_bar-8": BasicChartSpecs(
                        title="核心利润百分位数",
                        data_source="rolling_financial",
                        column="核心利润_normalized"
                    ),
                    "_basic_bar-9": BasicChartSpecs(
                        title="核心利润增速",
                        data_source="rolling_financial",
                        column="核心利润_yoy"
                    ),
                    "_upper_lower_dichotomy-7": QuadrantsChartSpecs(
                        title="核心利润率",
                        ul_column="核心利润率",
                        ul_data_source="rolling_financial",
                        ll_column="核心利润率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_basic_bar-10": BasicChartSpecs(
                        title="净利润",
                        data_source="rolling_financial",
                        column="净利润"
                    ),
                    "_basic_bar-11": BasicChartSpecs(
                        title="净利润百分位数",
                        data_source="rolling_financial",
                        column="净利润_normalized"
                    ),
                    "_basic_bar-12": BasicChartSpecs(
                        title="净利润增速",
                        data_source="rolling_financial",
                        column="净利润_yoy"
                    ),
                    "_basic_bar-13": BasicChartSpecs(
                        title="经营净利润",
                        data_source="rolling_financial",
                        column="经营净利润"
                    ),
                    "_basic_bar-14": BasicChartSpecs(
                        title="经营净利润百分位数",
                        data_source="rolling_financial",
                        column="经营净利润_normalized"
                    ),
                    "_basic_bar-15": BasicChartSpecs(
                        title="经营净利润增速",
                        data_source="rolling_financial",
                        column="经营净利润_yoy"
                    ),
                    "_upper_lower_dichotomy-8": QuadrantsChartSpecs(
                        title="经营性资产收益率",
                        ul_column="经营性资产收益率",
                        ul_data_source="rolling_financial",
                        ll_column="经营性资产收益率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-9": QuadrantsChartSpecs(
                        title="金融性资产收益率",
                        ul_column="金融性资产收益率",
                        ul_data_source="rolling_financial",
                        ll_column="金融性资产收益率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-10": QuadrantsChartSpecs(
                        title="核心利润获现率",
                        ul_column="核心利润获现率",
                        ul_data_source="rolling_financial",
                        ll_column="核心利润获现率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-11": QuadrantsChartSpecs(
                        title="收现比",
                        ul_column="收现比",
                        ul_data_source="rolling_financial",
                        ll_column="收现比_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-12": QuadrantsChartSpecs(
                        title="净现比",
                        ul_column="净现比",
                        ul_data_source="rolling_financial",
                        ll_column="净现比_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-13": QuadrantsChartSpecs(
                        title="经营活动产生的现金流量净额",
                        ul_column="经营活动产生的现金流量净额",
                        ul_data_source="rolling_financial",
                        ll_column="经营活动产生的现金流量净额_normalized",
                        ll_data_source="rolling_financial"
                    ),
                },
                "营运能力": {
                    "_basic_bar-1": BasicChartSpecs(
                        title="存货",
                        data_source="rolling_financial",
                        column="存货"
                    ),
                    "_basic_bar-2": BasicChartSpecs(
                        title="存货百分位数",
                        data_source="rolling_financial",
                        column="存货_normalized"
                    ),
                    "_basic_bar-3": BasicChartSpecs(
                        title="存货增速",
                        data_source="rolling_financial",
                        column="存货_yoy"
                    ),
                    "_upper_lower_dichotomy-1": QuadrantsChartSpecs(
                        title="现金转换周期",
                        ul_column="现金转换周期",
                        ul_data_source="rolling_financial",
                        ll_column="现金转换周期_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-2": QuadrantsChartSpecs(
                        title="营业周期",
                        ul_column="营业周期",
                        ul_data_source="rolling_financial",
                        ll_column="营业周期_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-3": QuadrantsChartSpecs(
                        title="存货周转天数",
                        ul_column="存货周转天数",
                        ul_data_source="rolling_financial",
                        ll_column="存货周转天数_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-4": QuadrantsChartSpecs(
                        title="应收票据及应收账款周转天数",
                        ul_column="应收票据及应收账款周转天数",
                        ul_data_source="rolling_financial",
                        ll_column="应收票据及应收账款周转天数_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-5": QuadrantsChartSpecs(
                        title="应付票据及应付账款周转天数",
                        ul_column="应付票据及应付账款周转天数",
                        ul_data_source="rolling_financial",
                        ll_column="应付票据及应付账款周转天数_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-6": QuadrantsChartSpecs(
                        title="总资产周转天数",
                        ul_column="总资产周转天数",
                        ul_data_source="rolling_financial",
                        ll_column="总资产周转天数_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-7": QuadrantsChartSpecs(
                        title="流动资产周转天数",
                        ul_column="流动资产周转天数",
                        ul_data_source="rolling_financial",
                        ll_column="流动资产周转天数_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-8": QuadrantsChartSpecs(
                        title="非流动资产周转天数",
                        ul_column="非流动资产周转天数",
                        ul_data_source="rolling_financial",
                        ll_column="非流动资产周转天数_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-9": QuadrantsChartSpecs(
                        title="固定资产周转天数",
                        ul_column="固定资产周转天数",
                        ul_data_source="rolling_financial",
                        ll_column="固定资产周转天数_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-10": QuadrantsChartSpecs(
                        title="上下游资金占用",
                        ul_column="上下游资金占用",
                        ul_data_source="rolling_financial",
                        ll_column="上下游资金占用_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-11": QuadrantsChartSpecs(
                        title="单位营收所需的经营性营运资本",
                        ul_column="单位营收所需的经营性营运资本",
                        ul_data_source="rolling_financial",
                        ll_column="单位营收所需的经营性营运资本_normalized",
                        ll_data_source="rolling_financial"
                    ),
                },
                "成长能力": {
                    "_basic_bar-10": BasicChartSpecs(
                        title="固定资产原值",
                        data_source="rolling_financial",
                        column="固定资产原值"
                    ),
                    "_basic_bar-11": BasicChartSpecs(
                        title="固定资产原值百分位数",
                        data_source="rolling_financial",
                        column="固定资产原值_normalized"
                    ),
                    "_basic_bar-12": BasicChartSpecs(
                        title="固定资产原值增速",
                        data_source="rolling_financial",
                        column="固定资产原值_yoy"
                    ),
                    "_basic_bar-1": BasicChartSpecs(
                        title="固定资产净额",
                        data_source="rolling_financial",
                        column="固定资产净额"
                    ),
                    "_basic_bar-2": BasicChartSpecs(
                        title="固定资产净额百分位数",
                        data_source="rolling_financial",
                        column="固定资产净额_normalized"
                    ),
                    "_basic_bar-3": BasicChartSpecs(
                        title="固定资产净额增速",
                        data_source="rolling_financial",
                        column="固定资产净额_yoy"
                    ),
                    "_upper_lower_dichotomy-1": QuadrantsChartSpecs(
                        title="固定资产原值推动力",
                        ul_column="固定资产原值推动力",
                        ul_data_source="rolling_financial",
                        ll_column="固定资产原值推动力_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-6": QuadrantsChartSpecs(
                        title="固定资产净值推动力",
                        ul_column="固定资产净值推动力",
                        ul_data_source="rolling_financial",
                        ll_column="固定资产净值推动力_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-2": QuadrantsChartSpecs(
                        title="扩张倍数",
                        ul_column="扩张倍数",
                        ul_data_source="rolling_financial",
                        ll_column="扩张倍数_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-3": QuadrantsChartSpecs(
                        title="收缩倍数",
                        ul_column="收缩倍数",
                        ul_data_source="rolling_financial",
                        ll_column="收缩倍数_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-4": QuadrantsChartSpecs(
                        title="内含增长率",
                        ul_column="内含增长率",
                        ul_data_source="rolling_financial",
                        ll_column="内含增长率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                    "_upper_lower_dichotomy-5": QuadrantsChartSpecs(
                        title="可持续增长率",
                        ul_column="可持续增长率",
                        ul_data_source="rolling_financial",
                        ll_column="可持续增长率_normalized",
                        ll_data_source="rolling_financial"
                    ),
                },
                "现金流": {
                    "_basic_bar-1": BasicChartSpecs(
                        title="企业自由现金流",
                        data_source="rolling_financial",
                        column="企业自由现金流"
                    ),
                    "_basic_bar-2": BasicChartSpecs(
                        title="股权自由现金流",
                        data_source="rolling_financial",
                        column="股权自由现金流"
                    ),
                    "_basic_bar-3": BasicChartSpecs(
                        title="无杠杆自由现金流",
                        data_source="rolling_financial",
                        column="无杠杆自由现金流"
                    ),
                    "_basic_bar-4": BasicChartSpecs(
                        title="杠杆自由现金流",
                        data_source="rolling_financial",
                        column="杠杆自由现金流"
                    ),
                },
                "估值": {
                    "_basic_bar-1": BasicChartSpecs(
                        title="市值",
                        data_source="valuation",
                        column="市值"
                    ),
                    "_upper_lower_dichotomy-1": QuadrantsChartSpecs(
                        title="市值分位数",
                        ul_column="市值_normalized",
                        ul_data_source="valuation",
                        ll_column="市值_rolling_normalized",
                        ll_data_source="valuation"
                    ),
                    "_basic_bar-2": BasicChartSpecs(
                        title="核心利润市盈率",
                        data_source="valuation",
                        column="核心利润市盈率"
                    ),
                    "_upper_lower_dichotomy-2": QuadrantsChartSpecs(
                        title="核心利润市盈率",
                        ul_column="核心利润市盈率_normalized",
                        ul_data_source="valuation",
                        ll_column="核心利润市盈率_rolling_normalized",
                        ll_data_source="valuation"
                    ),
                    "_basic_bar-3": BasicChartSpecs(
                        title="市净率",
                        data_source="valuation",
                        column="市净率"
                    ),
                    "_upper_lower_dichotomy-3": QuadrantsChartSpecs(
                        title="市净率",
                        ul_column="市净率_normalized",
                        ul_data_source="valuation",
                        ll_column="市净率_rolling_normalized",
                        ll_data_source="valuation"
                    ),
                    "_basic_bar-4": BasicChartSpecs(
                        title="市销率",
                        data_source="valuation",
                        column="市销率"
                    ),
                    "_upper_lower_dichotomy-4": QuadrantsChartSpecs(
                        title="市销率",
                        ul_column="市销率_normalized",
                        ul_data_source="valuation",
                        ll_column="市销率_rolling_normalized",
                        ll_data_source="valuation"
                    ),
                    "_basic_bar-5": BasicChartSpecs(
                        title="周期市盈率",
                        data_source="valuation",
                        column="周期市盈率"
                    ),
                    "_upper_lower_dichotomy-5": QuadrantsChartSpecs(
                        title="周期市盈率",
                        ul_column="周期市盈率_normalized",
                        ul_data_source="valuation",
                        ll_column="周期市盈率_rolling_normalized",
                        ll_data_source="valuation"
                    ),
                    "_basic_bar-6": BasicChartSpecs(
                        title="市盈增长率",
                        data_source="valuation",
                        column="市盈增长率"
                    ),
                    "_upper_lower_dichotomy-6": QuadrantsChartSpecs(
                        title="市盈增长率",
                        ul_column="市盈增长率_normalized",
                        ul_data_source="valuation",
                        ll_column="市盈增长率_rolling_normalized",
                        ll_data_source="valuation"
                    ),
                    "_basic_bar-7": BasicChartSpecs(
                        title="股息率",
                        data_source="valuation",
                        column="股息率"
                    ),
                    "_upper_lower_dichotomy-7": QuadrantsChartSpecs(
                        title="股息率",
                        ul_column="股息率_normalized",
                        ul_data_source="valuation",
                        ll_column="股息率_rolling_normalized",
                        ll_data_source="valuation"
                    ),
                    "_basic_bar-8": BasicChartSpecs(
                        title="核心利润实际收益率",
                        data_source="valuation",
                        column="核心利润实际收益率"
                    ),
                    "_upper_lower_dichotomy-8": QuadrantsChartSpecs(
                        title="核心利润实际收益率",
                        ul_column="核心利润实际收益率_normalized",
                        ul_data_source="valuation",
                        ll_column="核心利润实际收益率_rolling_normalized",
                        ll_data_source="valuation"
                    ),
                }
            }


#############################################################
class Quant:
    def __init__(self):
        # 财务模块
        self.finance = self.Finance()
        # 估值模块
        self.valuation = self.Valuation()
        # 量价模块
        self.kline = self.Kline()
        # 统计模块
        self.stat = self.Statistics()
        # 可视化模块
        self.visualization = self.Visualization()

    class Finance:
        def __init__(self):
            """
            盈利能力：衡量企业赚取利润的能力。
            运营能力：反映企业资产运营效率和管理水平。
            偿债能力：评估企业偿还债务的能力。
            成长能力：衡量企业未来发展潜力和增长能力。
            现金流量：分析企业现金流入和流出的情况。
            资本结构：反映企业资金来源和构成。
            费用控制：评估企业对各项费用的控制能力。
            资产质量：衡量企业资产的安全性和价值。
            税务管理：反映企业税务负担和管理效率。
            其他：未明确归类的指标，通常为基础数据或辅助分析指标。
            """
            # 基础报表
            self.basic_reports = {
                "资产负债表": [
                    "非流动资产合计", "少数股东权益", "应收票据", "应收账款", "实收资本",
                    "库存股", "存货", "所有者权益", "负债和所有者权益"
                ],
                "利润表": [
                    "利润总额", "资产减值损失", "信用减值损失", "归属于母公司所有者的净利润",
                    "净利润", "营业利润", "营业收入"
                ],
                "现金流量表": [
                    "投资活动现金流入小计", "投资活动现金流出小计", "收回投资所收到的现金",
                    "取得投资收益所收到的现金", "处置固定资产、无形资产和其他长期资产所收回的现金净额",
                    "收到的其他与投资活动有关的现金", "购建固定资产、无形资产和其他长期资产支付的现金",
                    "投资所支付的现金", "取得子公司及其他营业单位支付的现金净额", "支付的其他与投资活动有关的现金",
                    "投资活动产生的现金流量净额", "筹资活动现金流入小计", "筹资活动现金流出小计",
                    "筹资活动产生的现金流量净额", "吸收投资收到的现金", "取得借款收到的现金",
                    "发行债券收到的现金", "收到其他与筹资活动有关的现金", "偿还债务支付的现金",
                    "分配股利、利润或偿付利息所支付的现金", "支付其他与筹资活动有关的现金", "经营活动现金流入小计",
                    "经营活动现金流出小计", "经营活动产生的现金流量净额", "销售商品、提供劳务收到的现金",
                    "收到的税费返还", "收到的其他与经营活动有关的现金", "购买商品、接受劳务支付的现金",
                    "支付给职工以及为职工支付的现金", "支付的各项税费", "支付的其他与经营活动有关的现金"
                ]
            }
            # 财务分析
            self.financial_analysis = {
                "盈利能力": [
                    "毛利率", "核心利润率", "营业净利率", "核心利润净利率", "权益净利率",
                    "归母权益净利率", "总资产净利率", "净经营资产净利率", "经营净利率",
                    "销售利润率", "经营差异率", "杠杆贡献率", "核心利润占比", "经营性资产收益率",
                    "金融性资产收益率", "毛利", "投资利润", "杂项利润", "营业外收支净额",
                    "息税前利润", "息税前利润率", "核心利润"
                ],
                "运营能力": [
                    "存货周转天数", "应收票据及应收账款周转天数", "应付票据及应付账款周转天数",
                    "现金转换周期", "固定资产周转率", "总资产周转率", "单位营收所需的经营性营运资本",
                    "营业周期", "固定资产周转天数", "流动资产周转天数", "非流动资产周转天数",
                    "总资产周转天数", "净经营资产周转率", "固定资产原值推动力", "固定资产净值推动力",
                ],
                "偿债能力": [
                    "流动比率", "速动比率", "现金比率", "资产负债率", "有息负债率",
                    "债务资本化比率", "长期资本化比率", "利息保障倍数", "现金流量利息保障倍数",
                    "净财务杠杆", "现金流量比率"
                ],
                "成长能力": [
                    "内含增长率", "可持续增长率", "利润留存率", "股利支付率", "扩张倍数",
                    "收缩倍数"
                ],
                "现金流量": [
                    "经营现金流量", "债务现金流量", "股权现金流量", "核心利润获现率",
                    "收现比", "净现比", "企业自由现金流", "股权自由现金流",
                    "无杠杆自由现金流", "杠杆自由现金流",
                ],
                "资本结构": [
                    "金融性负债占比", "经营性负债占比", "股东入资占比", "利润留存占比",
                    "流动资产占比", "货币资金占比", "经营性资产占比", "金融性资产占比",
                    "应付款比率", "预收款比率", "上下游资金占用"
                ],
                "费用控制": [
                    "销售费用率", "管理费用率", "财务费用率", "研发费用率", "期间费用率",
                    "毛销差", "营业税金负担率"
                ],
                "资产质量": [
                    "信用减值损失净资产占比", "资产减值损失净资产占比", "减值损失核心利润占比"
                ],
                "税务管理": [
                    "实际税率"
                ],
                "其他": [
                    "经营性流动资产", "经营性流动负债", "经营性营运资本", "经营性长期资产",
                    "经营性长期负债", "净经营性长期资产", "净经营资产", "金融性负债",
                    "金融性资产", "净负债", "经营净利润", "金融净利润"
                ]
            }

    class Valuation:
        def __init__(self):
            # 基础指标
            self.basic_metrics = [
                "每股核心利润", "每股净资产", "每股销售额", "每股分红", "市值", "对数市值"
            ]
            # 衍生指标
            self.derived_metrics = [
                "市销率倒数", "市净率倒数", "核心利润盈利市值比", "周期市盈率倒数", "市盈增长率倒数",
                "股息率", "核心利润实际收益率", "实际收益率", "市现率倒数", "市净率"
            ]

    class Kline:
        def __init__(self):
            self.kline = {}

    class Statistics:
        def __init__(self):
            self.stats = ["同比", "滚动归一化"]

    class Visualization:
        def __init__(self):
            # page名
            self.pages_name = []
            # 图表配置
            self.pages_config = {}


#############################################################
class Factor:
    def __init__(self):
        # 财务模块
        self.finance = self.Finance()
        # 估值模块
        self.valuation = self.Valuation()
        # 量价模块
        self.kline = self.Kline()
        # 统计模块
        self.stat = self.Statistics()
        # 可视化模块
        self.visualization = self.Visualization()

    class Finance:
        def __init__(self):
            """
            盈利能力：衡量企业赚取利润的能力。
            运营能力：反映企业资产运营效率和管理水平。
            偿债能力：评估企业偿还债务的能力。
            成长能力：衡量企业未来发展潜力和增长能力。
            现金流量：分析企业现金流入和流出的情况。
            资本结构：反映企业资金来源和构成。
            费用控制：评估企业对各项费用的控制能力。
            资产质量：衡量企业资产的安全性和价值。
            税务管理：反映企业税务负担和管理效率。
            其他：未明确归类的指标，通常为基础数据或辅助分析指标。
            """
            # 基础报表
            self.basic_reports = {
                "资产负债表": [
                    "所有者权益"
                ],
                "利润表": [
                ],
                "现金流量表": [
                ]
            }
            # 财务分析
            self.financial_analysis = {
                "盈利能力": [
                ],
                "运营能力": [
                ],
                "偿债能力": [
                ],
                "成长能力": [
                ],
                "现金流量": [
                ],
                "资本结构": [
                ],
                "费用控制": [
                ],
                "资产质量": [
                ],
                "税务管理": [
                ],
                "其他": [
                ]
            }

    class Valuation:
        def __init__(self):
            # 基础指标
            self.basic_metrics = [
                "市值", "对数市值", "市净率"
            ]
            # 衍生指标
            self.derived_metrics = [
            ]

    class Kline:
        def __init__(self):
            self.kline = {
                # "k线中继形态": [1],
                # "市场贝塔": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "异常换手率": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "累加收益率": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "真实波幅": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "收盘价均线": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "成交量均线": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "换手率均线": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "真实波幅均线": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "收益率标准差": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "换手率标准差": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                # "真实波幅标准差": [
                #     0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                # ],
                "斜率": [
                    0.09, 0.17, 0.25, 0.5, 1, 1.5, 2
                ],
            }

    class Statistics:
        def __init__(self):
            self.stats = []

    class Visualization:
        def __init__(self):
            # page名
            self.pages_name = []
            # 图表配置
            self.pages_config = {}


#############################################################
class InventoryCycle:

    def __init__(self):
        # 财务模块
        self.finance = self.Finance()
        # 估值模块
        self.valuation = self.Valuation()
        # 量价模块
        self.kline = self.Kline()
        # 统计模块
        self.stat = self.Statistics()
        # 可视化模块
        self.visualization = self.Visualization()

    class Finance:
        def __init__(self):
            """
            盈利能力：衡量企业赚取利润的能力。
            运营能力：反映企业资产运营效率和管理水平。
            偿债能力：评估企业偿还债务的能力。
            成长能力：衡量企业未来发展潜力和增长能力。
            现金流量：分析企业现金流入和流出的情况。
            资本结构：反映企业资金来源和构成。
            费用控制：评估企业对各项费用的控制能力。
            资产质量：衡量企业资产的安全性和价值。
            税务管理：反映企业税务负担和管理效率。
            其他：未明确归类的指标，通常为基础数据或辅助分析指标。
            """
            # 基础报表
            self.basic_reports = {
                "资产负债表": [
                    "存货"
                ],
                "利润表": [
                    "营业收入", "净利润", "在建工程", "固定资产净额",
                ],
                "现金流量表": [
                    "购买商品、接受劳务支付的现金", "购建固定资产、无形资产和其他长期资产支付的现金",
                    "偿还债务支付的现金",
                ]
            }
            # 财务分析
            self.financial_analysis = {
                "盈利能力": ["营业收入(一阶差分)"],
                "运营能力": ["存货投资"],
                "偿债能力": ["资产负债率", "有息负债率"],
                "资产质量": ["固定资产原值"]
            }

    class Valuation:
        def __init__(self):
            # 基础指标
            self.basic_metrics = ["市值"]
            # 衍生指标
            self.derived_metrics = []

    class Kline:
        def __init__(self):
            self.kline = {}

    class Statistics:
        def __init__(self):
            self.stats = ["同比", "复合增长率", "归一化"]

    class Visualization:
        def __init__(self):
            # page名
            self.pages_name = [
                "库存周期"
            ]
            # 图表配置
            self.pages_config = {
                "库存周期": {
                    "_upper_lower_dichotomy-1": QuadrantsChartSpecs(
                        title="营业收入",
                        ul_data_source="rolling_financial",
                        ul_column="营业收入",
                        ll_data_source="rolling_financial",
                        ll_column="营业收入_yoy"
                    ),
                    "_upper_lower_dichotomy-2": QuadrantsChartSpecs(
                        title="净利润",
                        ul_data_source="rolling_financial",
                        ul_column="净利润",
                        ll_data_source="rolling_financial",
                        ll_column="净利润_yoy"
                    ),
                    "_upper_lower_dichotomy-3": QuadrantsChartSpecs(
                        title="存货",
                        ul_data_source="rolling_financial",
                        ul_column="存货",
                        ll_data_source="rolling_financial",
                        ll_column="存货_yoy"
                    ),
                    "_upper_lower_dichotomy-4": QuadrantsChartSpecs(
                        title="存货投资",
                        ul_data_source="rolling_financial",
                        ul_column="存货投资",
                        ll_data_source="rolling_financial",
                        ll_column="存货投资_yoy"
                    ),
                    "_upper_lower_dichotomy-5": QuadrantsChartSpecs(
                        title="在建工程",
                        ul_data_source="rolling_financial",
                        ul_column="在建工程",
                        ll_data_source="rolling_financial",
                        ll_column="在建工程_yoy"
                    ),
                    "_upper_lower_dichotomy-6": QuadrantsChartSpecs(
                        title="固定资产原值",
                        ul_data_source="rolling_financial",
                        ul_column="固定资产原值",
                        ll_data_source="rolling_financial",
                        ll_column="固定资产原值_yoy"
                    ),
                    "_upper_lower_dichotomy-7": QuadrantsChartSpecs(
                        title="资产负债率",
                        ul_data_source="rolling_financial",
                        ul_column="资产负债率",
                        ll_data_source="rolling_financial",
                        ll_column="资产负债率_yoy"
                    ),
                    "_upper_lower_dichotomy-8": QuadrantsChartSpecs(
                        title="有息负债率",
                        ul_data_source="rolling_financial",
                        ul_column="有息负债率",
                        ll_data_source="rolling_financial",
                        ll_column="有息负债率_yoy"
                    ),
                    "_upper_lower_dichotomy-9": QuadrantsChartSpecs(
                        title="购买商品",
                        ul_data_source="rolling_financial",
                        ul_column="购买商品、接受劳务支付的现金",
                        ll_data_source="rolling_financial",
                        ll_column="购买商品、接受劳务支付的现金_yoy"
                    ),
                    "_upper_lower_dichotomy-10": QuadrantsChartSpecs(
                        title="购建资产",
                        ul_data_source="rolling_financial",
                        ul_column="购建固定资产、无形资产和其他长期资产支付的现金",
                        ll_data_source="rolling_financial",
                        ll_column="购建固定资产、无形资产和其他长期资产支付的现金_yoy"
                    ),
                    "_upper_lower_dichotomy-11": QuadrantsChartSpecs(
                        title="偿还债务",
                        ul_data_source="rolling_financial",
                        ul_column="偿还债务支付的现金",
                        ll_data_source="rolling_financial",
                        ll_column="偿还债务支付的现金_yoy"
                    ),
                    "_upper_lower_dichotomy-12": QuadrantsChartSpecs(
                        title="库销比",
                        ul_data_source="rolling_financial",
                        ul_column="库销比",
                        ll_data_source="rolling_financial_normalized",
                        ll_column="库销比"
                    ),
                    # "_basic_bar-1": BasicChartConfig(
                    #     title="库销比",
                    #     data_source="rolling_financial",
                    #     column="库销比"
                    # ),
                    # "_basic_scatter-1": BasicChartConfig(
                    #     title="存货同比",
                    #     data_source="rolling_financial",
                    #     column="存货_yoy"
                    # ),
                }
            }
