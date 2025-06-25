import random
import time
import threading
import pandas as pd

from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttrader import XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant

from constant import *

class MiniQMTTrader:

    def __init__(
            self,
            mini_qmt_path: str,
            account_id: str,
            session_id: int | None = None,
            account_type: str = 'STOCK'
    ):
        """
        初始化MiniQMT交易类
        :param mini_qmt_path: MiniQMT客户端的安装路径
        :param account_id: 资金账号
        :param session_id: 会话ID，若不指定则随机生成（建议不同策略使用不同ID）
        :param account_type: 账户类型，默认为股票账户
        """
        self.mini_qmt_path = mini_qmt_path
        self.account_id = account_id
        self.session_id = session_id or random.randint(100000, 999999)
        self.account_type = account_type

        # 创建交易对象和账户对象
        self.xt_trader = XtQuantTrader(mini_qmt_path, self.session_id)
        self.account = StockAccount(account_id, account_type)

        # 初始化回调处理器
        self.callback = self.TraderCallback(self)
        self.xt_trader.register_callback(self.callback)

        # 启动交易连接
        self.xt_trader.start()
        connect_result = self.xt_trader.connect()
        if connect_result == 0:
            print(f"连接MiniQMT成功！会话ID: {self.session_id}, 资金账号: {self.account_id}")
        else:
            raise ConnectionError(f"连接失败，错误码: {connect_result}")

        # 订阅账户
        subscribe_result = self.xt_trader.subscribe(self.account)
        print(f"账户订阅结果: {subscribe_result}")

    class TraderCallback(XtQuantTraderCallback):
        """内部回调类，处理交易相关的事件"""
        def __init__(self, outer_instance):
            super().__init__()
            self.outer = outer_instance

        def on_disconnected(self):
            """连接断开回调（可自动重连）"""
            print("交易连接断开，尝试重连...")
            self.outer.xt_trader.reconnect()

        def on_stock_order(self, order):
            """委托状态更新回调"""
            print(
                f"[委托状态] 合约: {order.stock_code}, "
                f"状态: {order.order_status}, "
                f"数量: {order.order_volume}, "
                f"价格: {order.price}"
            )

        def on_stock_trade(self, trade):
            """成交回报回调"""
            print(
                f"[成交回报] 合约: {trade.stock_code}, "
                f"数量: {trade.traded_volume}, "
                f"价格: {trade.traded_price}, "
                f"方向: {trade.side}"
            )

        def on_order_error(self, order_error):
            """委托失败回调"""
            """
            委托失败 需要根据订单号 再次发送委托
            """
            print(f"[委托错误] 错误码: {order_error.error_id}, 信息: {order_error.error_msg}")


        def on_cancel_error(self, cancel_error):
            """撤单失败回调"""
            """
            撤单失败 需要根据订单号 再次发送撤单请求
            """
            print(f"[撤单错误] 错误码: {cancel_error.error_id}, 信息: {cancel_error.error_msg}")

    def subscribe_quote(
            self,
            stock_list: list[str],
            callback: callable,
            period: str = 'tick'
    ):
        """
        订阅行情数据
        :param stock_list: 股票代码列表，['SH', 'SZ']
        :param callback: 行情数据回调函数，若不指定则使用类内部默认处理
        :param period: 行情周期，可选'tick'(分笔)、'1m'(1分钟)等
        """
        xtdata.subscribe_quote(stock_list, period, callback=callback)

    def on_tick(self, data):
        """
        事件驱动处理方法
        """
        for code, tick in data.items():
            print(f"[自定义回调] {code} 最新价: {tick['lastPrice']}, 时间: {tick['time']}")

    def run(self):
        """启动行情监听（阻塞式）"""
        xtdata.run()

    def order_stock(
            self,
            stock_code,
            price,
            volume,
            order_type=xtconstant.STOCK_BUY,
            price_type=0
    ):
        """
        股票下单
        :param stock_code: 股票代码，如 '600519.SH'
        :param price: 价格（市价单可填0）
        :param volume: 数量（股）
        :param order_type: 交易方向，BUY/SELL（使用xtconstant常量）
        :param price_type: 价格类型，0=市价单，1=限价单
        :return: 订单ID（用于后续撤单或查询）
        """
        return self.xt_trader.order_stock(
            account=self.account,
            stock_code=stock_code,
            order_type=order_type,
            order_volume=volume,
            price_type=price_type,
            price=price
        )

    def cancel_order(self, order_id):
        """撤单操作"""
        return self.xt_trader.cancel_order_stock(self.account, order_id)

    def get_asset(self):
        """查询账户资产"""
        return self.xt_trader.query_stock_asset(self.account)

    def get_positions(self):
        """查询持仓"""
        return self.xt_trader.query_stock_positions(self.account)

    def get_orders(self):
        """查询当日委托"""
        return self.xt_trader.query_stock_orders(self.account)

    def get_trades(self):
        """查询当日成交"""
        return self.xt_trader.query_stock_trades(self.account)


if __name__ == "__main__":
    # 创建交易实例
    trader = MiniQMTTrader(TRADE_MINI_QMT_PATH, ACCOUNT_ID)

    # 订阅全市场数据，传入回调函数
    trader.subscribe_quote(
        ['SH', 'SZ'],
        callback=trader.on_tick,
        period="tick"
    )

    # 启动行情监听（在独立线程中运行）
    data_thread = threading.Thread(target=trader.run)
    data_thread.daemon = True
    data_thread.start()

    # 主线程模拟交易
    try:
        while True:
            # 检视账户情况与模型生成的标的池的差异，根据差异买入标的

            # 示例：市价买入100股贵州茅台
            order_id = trader.order_stock('600519.SH', 0, 100, xtconstant.STOCK_BUY, price_type=0)
            print(f"已发送市价买单，订单ID: {order_id}")
            time.sleep(60)  # 等待1分钟
    except KeyboardInterrupt:
        print("程序退出")