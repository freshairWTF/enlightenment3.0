"""xtquant策略测试"""
import random
import time
import threading
import pandas as pd
from queue import Queue

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
            account_type: str = 'STOCK',
            interval: int = 10,
            queue_max_size: int = 10
    ):
        """
        初始化MiniQMT交易类
        :param mini_qmt_path: MiniQMT客户端的安装路径
        :param account_id: 资金账号
        :param session_id: 会话ID，若不指定则随机生成（建议不同策略使用不同ID）
        :param account_type: 账户类型，默认为股票账户
        :param interval: 定时器间隔时间
        :param queue_max_size: 队列最大规模
        """
        self.mini_qmt_path = mini_qmt_path
        self.account_id = account_id
        self.session_id = session_id or random.randint(100000, 999999)
        self.account_type = account_type

        # 创建交易对象和账户对象
        self.xt_trader = XtQuantTrader(mini_qmt_path, self.session_id)
        self.account = StockAccount(account_id, account_type)

        # 创建交易回调对象，并声明接受回调
        self.callback = self.TraderCallback(self)
        self.xt_trader.register_callback(self.callback)

        # 启动交易线程
        self.xt_trader.start()
        # 建立交易连接
        connect_result = self.xt_trader.connect()
        if connect_result == 0:
            print(f"连接MiniQMT成功！会话ID: {self.session_id}, 资金账号: {self.account_id}")
        else:
            raise ConnectionError(f"交易连接失败，错误码: {connect_result}")

        # 对交易回调进行订阅，订阅后可以收到交易主推
        subscribe_result = self.xt_trader.subscribe(self.account)
        if subscribe_result == 0:
            print(f"交易回调订阅成功: {subscribe_result}")
        else:
            raise ConnectionError(f"交易回调订阅失败: {subscribe_result}")

        # 创建定时器
        self.timer = self.RecurringTimer(interval=interval, task=self.on_timer)

        # 创建行情队列
        self.data_queue = Queue(maxsize=queue_max_size)

    # ---------------------------------------------
    # 操作方法 订阅/监听行情、k线推送处理、发送订单/撤单
    # ---------------------------------------------
    def subscribe_whole_quote(
            self,
            callback: callable,
    ):
        """
        订阅全推行情数据
        :param callback: 行情数据回调函数，若不指定则使用类内部默认处理
        """
        return xtdata.subscribe_whole_quote(['SH', 'SZ'], callback=callback)

    def run(self):
        """启动行情监听（阻塞式）"""
        xtdata.run()

    def on_tick(self, data):
        """
        事件驱动处理方法:
            -1 接受、存储最新行情数据
        """
        self.data_queue.put(data)

    def on_timer(self):
        """
        定时器回调方法
            -1 查询策略生成的买卖数据 -> 执行订单买卖（sleep控制多个订单生成频率）
            -2 查询订单回调 -> 根据订单状态做进一步处理
        """
        """
        xtconstant.ORDER_UNREPORTED	48	未报
        xtconstant.ORDER_WAIT_REPORTING	49	待报
        xtconstant.ORDER_REPORTED	50	已报
        xtconstant.ORDER_REPORTED_CANCEL	51	已报待撤
        xtconstant.ORDER_PARTSUCC_CANCEL	52	部成待撤
        xtconstant.ORDER_PART_CANCEL	53	部撤（已经有一部分成交，剩下的已经撤单）
        xtconstant.ORDER_CANCELED	54	已撤
        xtconstant.ORDER_PART_SUCC	55	部成（已经有一部分成交，剩下的待成交）
        xtconstant.ORDER_SUCCEEDED	56	已成
        xtconstant.ORDER_JUNK	57	废单
        xtconstant.ORDER_UNKNOWN	255	未知
        
        订单根据订单编号进行管理 
        核心在于成交 部成 -> 撤单 -> 部撤/已撤 -> 发送新订单
        废单与未知需要 特别警示 人工排查问题
        """
        # ----------------------------------
        # 旧订单处理
        # ----------------------------------
        # 查询订单状态

        # 若还未执行就撤单

        # ----------------------------------
        # 新订单处理
        # ----------------------------------
        # 查询买卖数据（生成的买卖数据需要带有时间戳，令交易端可以确认是否执行过，）

        #
        last_data = list(self.data_queue.queue)[-1]

    def order_stock(
            self,
            stock_code,
            price,
            volume,
            order_type: str = "buy",
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
            order_type=xtconstant.STOCK_BUY if order_type == "buy" else xtconstant.STOCK_SELL,
            order_volume=volume,
            price_type=price_type,
            price=price
        )

    def cancel_order(
            self,
            order_id
    ):
        """
        股票撤单
        :param order_id: 订单编号
        :return: 是否成功发出撤单指令，0: 成功, -1: 表示撤单失败
        """
        return self.xt_trader.cancel_order_stock(
            account=self.account,
            order_id=order_id
        )

    # ---------------------------------------------
    # 查询方法
    # ---------------------------------------------
    def get_asset(self):
        """查询账户资产"""
        return self.xt_trader.query_stock_asset(self.account)

    def get_orders(self):
        """查询当日委托"""
        return self.xt_trader.query_stock_orders(self.account)

    def get_trades(self):
        """查询当日成交"""
        return self.xt_trader.query_stock_trades(self.account)

    def get_positions(self):
        """查询持仓"""
        return self.xt_trader.query_stock_positions(self.account)

    # ---------------------------------------------
    # 事件回调
    # ---------------------------------------------
    class TraderCallback(XtQuantTraderCallback):
        """内部回调类，处理交易相关的事件"""

        def __init__(self, outer_instance):
            super().__init__()
            self.outer = outer_instance

        def on_disconnected(self):
            """连接断开回调（可自动重连）"""
            print("交易连接断开，尝试重连...")
            self.outer.xt_trader.reconnect()

        def on_account_status(self, status):
            """
            账号状态信息推送
            :param status: XtAccountStatus 对象
            """
            print("账户信息推送")
            print(
                f"账户ID: {status.account_id}, "
                f"账户类型: {status.account_type}, "
                f"账户状态: {status.status}, "
            )
            if status.status != 0:
                print("账户状态异常！")

        def on_stock_order(self, order):
            """
            委托状态更新回调
            :param order: XtOrder对象
            """
            print("订单信息推送")
            print(
                f"订单编号: {order.order_id}, "
                f"报单时间: {order.order_time}, "
                f"证券代码: {order.stock_code}, "
                f"委托类型: {order.order_type}, "
                f"委托数量: {order.order_volume}, "
                f"委托价格: {order.price}"
                f"委托状态: {order.order_status}, "
                f"成交均价: {order.traded_price}, "
                f"成交数量: {order.traded_volume}"
            )

        def on_stock_trade(self, trade):
            """
            成交回报回调
            :param trade: XtTrade对象
            traded_id	str	成交编号
            traded_time	int	成交时间
            traded_price	float	成交均价
            traded_volume	int	成交数量
            traded_amount	float	成交金额
            order_id	int	订单编号
            """
            print("成交信息推送")
            print(
                f"成交编号: {trade.traded_id}, "
                f"证券代码: {trade.stock_code}, "
                f"成交时间: {trade.traded_time}, "
                f"成交均价: {trade.traded_price}, "
                f"成交数量: {trade.traded_volume}, "
                f"成交金额: {trade.traded_amount}, "
                f"订单编号: {trade.order_id}, "
            )

        def on_order_error(self, order_error):
            """
            委托失败回调
            :param order_error:XtOrderError 对象
            """
            print("订单委托失败")
            print(
                f"订单编号: {order_error.order_id}, "
                f"下单失败错误码: {order_error.error_id}, "
                f"下单失败具体信息: {order_error.error_msg}"
            )

        def on_cancel_error(self, cancel_error):
            """
            撤单失败回调
            :param cancel_error: XtCancelError 对象
            """
            print("撤单失败")
            print(
                f"订单编号: {cancel_error.order_id}, "
                f"撤单失败错误码: {cancel_error.error_id}, "
                f"撤单失败具体信息: {cancel_error.error_msg}"
            )

    # ---------------------------------------------
    # 定时器
    # ---------------------------------------------
    class RecurringTimer:
        def __init__(self, interval, task):
            """
            :param interval: 间隔时间
            :param task: 定时器触发时执行的核心任务
            """
            self.interval = interval
            self.task = task
            self.timer = None
            self._active = False

        def start(self):
            """首次创建、启动定时器"""
            self._active = True
            self._schedule()

        def _schedule(self):
            """创建启动定时器"""
            if self._active:
                self.timer = threading.Timer(self.interval, self._run)
                self.timer.daemon = True
                self.timer.start()

        def _run(self):
            """执行回调函数，再自动创建、启动新定时器"""
            self.task()             # 执行任务
            self._schedule()        # 自动调度下一次

        def stop(self):
            """停止定时器"""
            self._active = False
            if self.timer:
                self.timer.cancel()


if __name__ == "__main__":
    # 创建交易实例
    TRADE_MINI_QMT_PATH = "D:\国金QMT交易端模拟\\userdata_mini"
    ACCOUNT_ID = "40069146"
    trader = MiniQMTTrader(
        TRADE_MINI_QMT_PATH,
        ACCOUNT_ID,
        session_id=100000,
        interval=60,
        queue_max_size=10
    )

    # 订阅全市场数据，传入回调函数
    subscribe_code = trader.subscribe_whole_quote(callback=trader.on_tick)
    if subscribe_code == -1:
        raise ConnectionError(f"订阅全推行情数据失败: {subscribe_code}")

    # 启动行情监听（独立守护线程）
    data_thread = threading.Thread(target=trader.run)
    data_thread.daemon = True
    data_thread.start()

    """
    print 使用logger
    """

    # 行情结束，查询当日总体情况
    asset = trader.get_asset()
    orders = trader.get_orders()
    trades = trader.get_trades()
    position = trader.get_positions()
