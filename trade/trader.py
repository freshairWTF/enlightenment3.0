"""xtquant策略测试"""
import os
import sys
import time
import json
import threading

from loguru import logger
from datetime import datetime, time as time_class

from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttrader import XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant

from constant.path_config import DataPATH
from order_mgmt import OrderManager


xtdata.enable_hello = False

"""
9点20之后才能接受数据
9点25-9点30也能有行情
一旦策略订单发送完毕，不再重复记录
"""


##############################################################
class MiniQMTTrader:
    """
    线程管理：
        -1 定时器
        -2 行情监听（阻塞）
        -3 订单管理
    """

    # 订单管理类属性
    ORDER_MAX_RETRY = 3                 # 最大重试次数
    ORDER_MAX_WAIT_TIME = 60            # 最大等待时间(秒)

    HEARTBEAT_MAX_RETRY = 3             # 心跳检测最大重试次数

    def __init__(
            self,
            mini_qmt_path: str,
            account_id: str,
            session_id: int = 100,
            account_type: str = 'STOCK',
            timer_interval: int = 10,
            heartbeat_interval: int = 300
    ):
        """
        初始化MiniQMT交易类
        :param mini_qmt_path: MiniQMT客户端的安装路径
        :param account_id: 资金账号
        :param session_id: 会话ID
        :param account_type: 账户类型，默认为股票账户
        :param timer_interval: 定时器间隔时间
        :param heartbeat_interval: 心跳检测间隔时间
        """
        self.mini_qmt_path = mini_qmt_path
        self.account_id = account_id
        self.session_id = session_id
        self.account_type = account_type
        self.last_data = {}                                     # 行情字典（用于交易）
        self.subscribe_code = None                              # 行情订阅号
        self.data_start_time = time_class(9, 20)   # 行情数据开始时间(9:20)
        self.trading_end_time = time_class(15, 0)  # 交易结束时间(15:00)

        # -----------------------------
        # 线程管理
        # -----------------------------
        self.main_thread_blocker = threading.Event()                            # 主线程阻塞器
        self.market_thread = None                                               # 行情监听线程
        self.timer = self.RepeatingTimer(self, timer_interval, self.on_timer)   # 定时器

        # -----------------------------
        # 日志
        # -----------------------------
        self._set_logger()
        self.trade_logger = logger.bind(func="trade")
        self.market_logger = logger.bind(func="market")

        # -----------------------------
        # 创建xtquant对象
        # -----------------------------
        # -1 交易对象
        self.xt_trader = XtQuantTrader(mini_qmt_path, self.session_id)
        # -2 账户对象
        self.account = StockAccount(account_id, account_type)
        # -3 交易回调对象，并声明接受回调
        self.callback = self.TraderCallback(self)
        self.xt_trader.register_callback(self.callback)

        # -----------------------------
        # 启动交易线程
        # -----------------------------
        self.xt_trader.start()
        # 建立交易连接
        connect_result = self.xt_trader.connect()
        if connect_result == 0:
            self.trade_logger.success(f"连接MiniQMT成功！会话ID: {self.session_id}, 资金账号: {self.account_id}")
        else:
            self.trade_logger.error(f"交易连接失败，错误码: {connect_result}")
            raise ConnectionError(f"交易连接失败，错误码: {connect_result}")

        # -----------------------------
        # 交易
        # -----------------------------
        subscribe_result = self.xt_trader.subscribe(self.account)
        if subscribe_result == 0:
            self.trade_logger.success(f"交易回调订阅成功: {subscribe_result}")
        else:
            self.trade_logger.error(f"交易回调订阅失败: {subscribe_result}")
            raise ConnectionError(f"交易回调订阅失败: {subscribe_result}")

        # -----------------------------
        # 订单
        # -----------------------------
        # -1 待处理订单
        self.pending_orders = self._get_pending_order_btw_signals_and_position(
            DataPATH.STRATEGIC_TRADING_BOOK
        )
        # -2 独立超时检查线程
        self.order_manager = OrderManager()
        self.timeout_checker = threading.Thread(
            target=self._check_order_timeout,
            daemon=True
        )
        self.timeout_checker.start()

        # -----------------------------
        # 心跳检测
        # -----------------------------
        self.last_heartbeat = datetime.now()                            # 上次心跳时间
        self.heartbeat_interval = heartbeat_interval                    # 心跳检测间隔(秒)

    # ----------------------------------------------
    # 初始化 方法
    # ----------------------------------------------
    @staticmethod
    def _set_logger() -> None:
        """设置日志"""
        # 移除默认处理器
        logger.remove()

        # --------------------------------
        # -1 交易日志（trade.log）
        # --------------------------------
        logger.add(
            "trade.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            filter=lambda record: record["extra"].get("func") == "trade",
            rotation="00:00",
            retention="365 days",
            enqueue=True,
            level="INFO",
            backtrace=True
        )
        # --------------------------------
        # -2 市场行情日志（market.log）
        # --------------------------------
        logger.add(
            "market.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            filter=lambda record: record["extra"].get("func") == "market",
            rotation="100 MB",
            retention="7 days",
            enqueue=True,
            level="INFO",
            backtrace=True
        )
        # --------------------------------
        # -3 错误日志（error.log）
        # --------------------------------
        logger.add(
            "error.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            filter=lambda record: record["level"].name == "ERROR",
            rotation="00:00",
            retention="365 days",
            enqueue=True,
            level="ERROR",
            backtrace=True
        )
        # --------------------------------
        # 4. 控制台输出配置（排除market日志）
        # --------------------------------
        logger.add(
            sink=sys.stdout,  # 输出到控制台
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
            filter=lambda record: record["extra"].get("func") != "market",  # 关键过滤：跳过market日志[6,7](@ref)
            level="INFO"
        )

    def _get_pending_order_btw_signals_and_position(
            self,
            signal_file_path: str
    ) -> list[dict]:
        """
        根据交易信号文件生成订单集合
            -1 读取JSON交易信号
            -2 查询当前账户持仓
            -3 对比信号与持仓生成买卖订单
            -4 校验价格是否在涨跌停范围内
        :param signal_file_path: 交易信号JSON文件路径
        :return: 生成的订单列表
        """
        # -1 读取交易信号文件
        if not os.path.exists(signal_file_path):
            self.trade_logger.error(f"交易信号文件不存在: {signal_file_path}")
            raise FileNotFoundError(f"交易信号文件不存在: {signal_file_path}")
        with open(signal_file_path, 'r', encoding='utf-8') as f:
            try:
                signals = json.load(f)
                signals = self.add_market_suffix(signals)
                self.trade_logger.success(f"成功读取交易信号: {len(signals)}条")
            except json.JSONDecodeError as e_:
                self.trade_logger.error(f"JSON文件解析错误: {str(e_)}")
                raise ValueError(f"JSON文件解析错误: {str(e_)}") from e_

        # -2 查询当前持仓
        positions = self.get_positions()
        position_dict = {pos.stock_code: pos.volume for pos in positions}
        self.trade_logger.info(f"当前持仓数量: {len(position_dict)}只股票")

        # -3 生成待处理订单字典（根据信号与持仓的差）
        pending_order = []
        for signal in signals:
            # -1 信号买入参数
            stock_code = signal.get("stock_code", "")
            target_volume = signal.get("volume", 0)
            if not stock_code or target_volume < 0:
                self.trade_logger.error(f"无效信号: 股票代码: {stock_code} | 目标数量: {target_volume}")
                continue
            # -2 当前持仓股数
            pos_volume = position_dict.get(stock_code, 0)
            # -3 信号与持仓差值
            volume_diff = target_volume - pos_volume
            # -4 生成实际买卖信号
            if volume_diff == 0:
                continue
            else:
                direct, volume = "buy" if volume_diff > 0 else "sell", abs(volume_diff)
                pending_order.append({
                    "stock_code": stock_code,
                    "order_type": direct,
                    "volume": volume,
                    "status": "PENDING",     # PENDING/SENT/CONFIRMED/FAILED
                    "retry_count": 0
                })
                self.trade_logger.info(f"生成待处理订单 -> stock_code: {stock_code} | order_type: {direct} | volume: {volume}")

        return pending_order

    def add_market_suffix(
            self,
            order: list[dict]
    ) -> list[dict]:
        """
        为股票订单添加交易所市场标识后缀
        :param order: 订单列表，每个订单是包含stock_code和order_type的字典
        :return: 添加了正确市场后缀的新订单列表
        """
        suffix_orders = []

        for order in order:
            code = order["股票代码"]

            # 根据股票代码开头确定交易所标识
            if code.startswith(('600', '601', '603', '605', '688')):
                suffix = '.SH'                                          # 上交所标识
            elif code.startswith(('000', '001', '002', '003')):
                suffix = '.SZ'                                          # 深交所主板标识
            elif code.startswith(('300', '301')):
                suffix = '.SZ'                                          # 深交所创业板标识
            elif code.startswith(('8', '9')):
                suffix = '.BJ'                                          # 北交所标识
            else:
                suffix = ''                                             # 未知类型保持原样
                self.trade_logger.error(f"添加市场标识，出现未知交易所代码: {code}")

            # 创建带后缀的新订单
            suffix_orders.append({
                "stock_code": code + suffix,
                "volume": order["买入股数"]
            })

        return suffix_orders

    # ---------------------------------------------
    # 操作方法 订阅/监听行情、k线推送处理、发送订单/撤单
    # ---------------------------------------------
    def subscribe_whole_quote(
            self,
            callback: callable,
    ) -> None:
        """
        订阅全推行情数据
        :param callback: 行情数据回调函数，若不指定则使用类内部默认处理
        """
        subscribe_code = xtdata.subscribe_whole_quote(['SH', 'SZ'], callback=callback)
        if subscribe_code == -1:
            self.market_logger.error(f"订阅全推行情数据失败: {subscribe_code}")
            raise ConnectionError(f"订阅全推行情数据失败: {subscribe_code}")
        else:
            self.market_logger.success(f"订阅全推行情数据成功: {subscribe_code}")

    def run(self) -> None:
        """
        -1 启动定时器
        -2 订阅行情
        -3 启动行情监听（非阻塞）
        """
        self.trade_logger.info(f"启动定时器 | 行情监听（阻塞）")

        # -1 启动定时器
        self.timer.start()

        # -2 订阅行情数据
        self.subscribe_code = xtdata.subscribe_whole_quote(['SH', 'SZ'], callback=self.on_tick)
        if self.subscribe_code == -1:
            self.market_logger.error(f"订阅全推行情数据失败: {self.subscribe_code}")
            raise ConnectionError(f"订阅全推行情数据失败: {self.subscribe_code}")
        else:
            self.market_logger.success(f"订阅全推行情数据成功: {self.subscribe_code}")

        # -3 启动行情监听线程（非阻塞/守护线程）
        self.market_thread = threading.Thread(target=xtdata.run, daemon=True)
        self.market_thread.start()

    def block_until_end(self) -> None:
        """
        主线程阻塞直到交易结束
        """
        self.trade_logger.info("主线程进入阻塞状态，等待交易结束...")

        try:
            # 主线程阻塞，直到交易结束时间或收到关闭信号
            while True:
                current_time = datetime.now().time()
                # 检查是否到达交易结束时间(15:00)
                if current_time >= self.trading_end_time:
                    self.trade_logger.info("到达交易结束时间(15:00)，准备关闭系统")
                    break
                # 每60秒检查一次
                time.sleep(60)
        except KeyboardInterrupt:
            self.trade_logger.warning("用户中断，准备关闭系统")

    def stop(self) -> None:
        """
        停止交易系统（线程安全）
        """
        self.trade_logger.info("开始关闭交易系统...")

        # -1 停止定时器
        self.timer.cancel()

        # -2 取消行情订阅
        if self.subscribe_code is not None:
            xtdata.unsubscribe_quote(self.subscribe_code)
            self.subscribe_code = None
            self.market_logger.success("已取消行情订阅")

        # -3 断开交易连接
        self.xt_trader.stop()
        self.trade_logger.success("交易系统已完全停止")

    def on_tick(self, data):
        """
        事件驱动处理方法: 仅用于接收、存储最新行情数据
        """
        current_time = datetime.now().time()
        if current_time > self.data_start_time:
            self.market_logger.info(f"接收到行情推送")
            self.last_data.update(data)

    def on_timer(self) -> None:
        """
        定时器回调方法
        """
        # -1 心跳检测
        self.heartbeat_detection()

        # -2 发送策略订单
        self.send_pending_order()

        # -3 撤单失败，再次撤单
        self._retry_failed_orders()

    def order_stock(
            self,
            stock_code: str,
            price: float,
            volume: int,
            order_type: str = "buy",
            price_type=1
    ) -> int:
        """
        股票下单
        :param stock_code: 股票代码，如 '600519.SH'
        :param price: 价格（市价单可填0）
        :param volume: 数量（股）
        :param order_type: 交易方向，BUY/SELL（使用xtconstant常量）
        :param price_type: 价格类型，0=市价单，1=限价单
            PS: 模拟盘不支持市价
        :return: 订单ID（用于后续撤单或查询）
        """
        self.trade_logger.info(
            f"发送订单: stock_code: {stock_code} | price: {price} | volume: {volume} | "
            f"order_type: {order_type} | price_type: {price_type}"
        )

        order_id = self.xt_trader.order_stock(
            account=self.account,
            stock_code=stock_code,
            order_type=xtconstant.STOCK_BUY if order_type == "buy" else xtconstant.STOCK_SELL,
            order_volume=volume,
            price_type=price_type,
            price=price
        )
        if order_id == -1:
            self.trade_logger.error(f"订单发送失败")
        else:
            self.trade_logger.success(f"订单发送成功 -> 订单ID: {order_id}")

        return order_id

    def order_stock_async(
            self,
            stock_code: str,
            price: float,
            volume: int,
            order_type: str = "buy",
            price_type: int = 1
    ) -> int:
        """
        异步股票下单（非阻塞）
        :param stock_code: 股票代码
        :param price: 价格（市价单可填0）
        :param volume: 数量
        :param order_type: 交易方向，buy/sell
        :param price_type: 价格类型，0=市价单，1=限价单
        :return: 订单ID（用于后续撤单或查询）
        """
        self.trade_logger.info(
            f"发送异步订单: stock_code={stock_code}, price={price}, "
            f"volume={volume}, type={order_type}, price_type={price_type}"
        )

        order_id = self.xt_trader.order_stock_async(
            account=self.account,
            stock_code=stock_code,
            order_type=xtconstant.STOCK_BUY if order_type == "buy" else xtconstant.STOCK_SELL,
            order_volume=volume,
            price_type=xtconstant.FIX_PRICE,
            price=price,
        )

        if order_id == -1:
            self.trade_logger.error("异步订单发送失败")
        else:
            self.trade_logger.success(f"异步订单已提交，ID: {order_id}")

        return order_id

    def cancel_order(
            self,
            order_id
    ) -> int:
        """
        股票撤单
        :param order_id: 订单编号
        :return: 是否成功发出撤单指令，0: 成功, -1: 表示撤单失败
        """
        self.trade_logger.info(f"撤单: order_id: {order_id}")

        resp = self.xt_trader.cancel_order_stock(
            account=self.account,
            order_id=order_id
        )
        if resp == -1:
            self.trade_logger.error(f"撤单请求发送失败")
        else:
            self.trade_logger.success(f"撤单请求发送成功 -> 订单ID: {order_id}")

        return resp

    def cancel_order_async(
            self,
            order_id
    ) -> int:
        """
        异步股票撤单（非阻塞）
        :param order_id: 订单编号
        :return: 是否成功发出撤单指令，0: 成功, -1: 表示撤单失败
        """
        self.trade_logger.info(f"撤单: order_id: {order_id}")

        resp = self.xt_trader.cancel_order_stock_async(
            account=self.account,
            order_id=order_id
        )
        if resp == -1:
            self.trade_logger.error(f"异步撤单请求发送失败")
        else:
            self.trade_logger.success(f"异步撤单请求发送成功 -> 订单ID: {order_id}")

        return resp

    # ---------------------------------------------
    # 订单管理
    # ---------------------------------------------
    def send_pending_order(self) -> None:
        """
        发送策略订单
        步骤：
            -1 检查是否有待发送订单
            -2 获取实时行情数据
            -3 生成限价单价格（考虑涨跌停限制）
            -4 批量发送订单
            -5 转移已发送订单到order_dict
        """
        if not self.pending_orders or not self.last_data:
            return
        self.trade_logger.info(f"开始发送策略订单，待处理订单数: {len(self.pending_orders)}")

        # 遍历并发送订单
        for order in self.pending_orders:
            # 不再重复发送: -1 已成功发送的订单 -2 多次发送失败的订单
            if (
                    order["status"] == "SENT" or
                    (order["status"] == "FAILED" and order["retry_count"] > self.ORDER_MAX_RETRY)
            ):
                continue

            # -1 待处理订单数据
            stock_code = order["stock_code"]
            order_type = order["order_type"]
            volume = order["volume"]
            if stock_code not in self.last_data.keys():
                self.trade_logger.warning(f"尚未监听到行情数据: {stock_code}")
                continue

            # -2 生成委托价格：使用五档行情第一档数据
            if order_type == "buy":
                ask_price = self.last_data[stock_code].get("askPrice", [])
                price = ask_price[0] if ask_price else self.last_data[stock_code]["lastPrice"] + 0.01
            else:
                bid_price = self.last_data[stock_code].get("bidPrice", [])
                price = bid_price[0] if bid_price else self.last_data[stock_code]["lastPrice"] - 0.01

            # -3 发送订单
            order_id = self.order_stock_async(
                stock_code=stock_code,
                price=round(price, 2),
                volume=volume,
                order_type=order_type,
                price_type=1
            )

            # -4 更新订单状态（PENDING/SENT/FAILED）
            if order_id == -1:
                order["status"] = "FAILED"
                order["retry_count"] += 1
            else:
                order["status"] = "SENT"

            # -5 日志记录
            if order["retry_count"] > self.ORDER_MAX_RETRY:
                self.trade_logger.error(
                    f"策略订单多次发送失败: stock_code: {stock_code} | price: {round(price, 2)} | "
                    f"volume: {volume} | order_type: {order_type}"
                )

        # -6 清理已发送订单
        self.pending_orders = [
            order for order in self.pending_orders
            if not (
                    order["status"] == "SENT"
                    or (order["status"] == "FAILED" and order["retry_count"] > self.ORDER_MAX_RETRY)
            )
        ]

    def _check_order_timeout(self) -> None:
        """
        独立线程检查订单超时
            -1 超时撤单
            -2 清理终态订单
        """
        while True:
            try:
                # 每k秒检查一次
                time.sleep(self.ORDER_MAX_WAIT_TIME)

                # -1 获取超时订单
                timeout_orders = self.order_manager.get_timeout_orders(
                    timeout_sec=self.ORDER_MAX_WAIT_TIME
                )

                # -2 触发撤单
                for order in timeout_orders:
                    self.trade_logger.warning(
                        f"订单超时未成交 | ID: {order['order_id']} | 尝试撤单"
                    )
                    self.order_manager.mark_canceling(order["order_id"])
                    self.cancel_order_async(order["order_id"])

                # -3 清理终态订单
                self.order_manager.cleanup_finalized()

            except Exception as e_:
                self.trade_logger.error(f"超时检查异常: {str(e_)}")

    def _retry_failed_orders(self) -> None:
        """重试失败订单（非即时敏感操作）"""
        with self.order_manager.lock:
            for order in self.order_manager.orders.values():
                # 只处理撤单失败的订单
                if order["status"] == "FAILED_CANCEL":
                    # 满足重试条件
                    if order["cancel_retry"] <= self.ORDER_MAX_RETRY:
                        self.trade_logger.info(
                            f"重试撤单 | ID: {order['order_id']}"
                        )
                        self.cancel_order_async(order["order_id"])

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
    # 心跳检测
    # ---------------------------------------------
    def heartbeat_detection(self):
        """心跳检测"""
        time_diff = (datetime.now() - self.last_heartbeat).seconds

        if time_diff > self.heartbeat_interval:
            self.trade_logger.info(f"心跳检测开始，间隔时间: {time_diff}")
            if not self._check_connection():
                self.trade_logger.error("连接验证失败，触发断连处理")
                self.callback.on_disconnected()
            # 更新心跳时间戳
            self.last_heartbeat = datetime.now()
            self.trade_logger.info(f"心跳检测正常，重新计时: {self.last_heartbeat}")

    def _check_connection(self) -> bool:
        """
        主动验证交易连接是否有效
        实现原理：发送轻量级查询测试连接状态[2,6](@ref)
        """
        try:
            # 尝试查询账户资产（轻量级操作）
            asset_ = self.xt_trader.query_stock_asset(self.account)

            # 成功获取数据表示连接正常
            if asset_ is not None:
                return True
            return False
        except Exception as e_:
            self.trade_logger.error(f"交易连接验证失败: {e_}")
            return False

    # ---------------------------------------------
    # 事件回调
    # ---------------------------------------------
    class TraderCallback(XtQuantTraderCallback):
        """内部回调类，处理交易相关的事件"""

        def __init__(self, outer_instance):
            super().__init__()
            self.outer = outer_instance

        def on_disconnected(self):
            """连接断开回调（可自动重连，最多3次）"""
            self.outer.trade_logger.warming("交易连接断开，尝试重连...")
            for _ in range(self.outer.HEARTBEAT_MAX_RETRY):
                if self.outer.xt_trader.reconnect() == 0:
                    self.outer.xt_trader.subscribe(self.outer.account)
                    return
                time.sleep(5)
            self.outer.trade_logger.error("重连失败，请检查网络")
            raise ConnectionError("重连失败，请检查网络")

        def on_account_status(self, status):
            """
            账号状态信息推送
            :param status: XtAccountStatus 对象
            """
            self.outer.trade_logger.info(
                f"账户信息推送, "
                f"账户ID: {status.account_id}, "
                f"账户类型: {status.account_type}, "
                f"账户状态: {status.status}, "
            )
            if status.status != 0:
                self.outer.trade_logger.error("！！！账户状态异常！！！")

        def on_stock_order(self, order):
            """
            委托状态更新回调
            :param order: XtOrder对象
            """
            self.outer.trade_logger.info(
                f"订单 信息推送: 订单ID={order.order_id}, 股票代码={order.stock_code}, "
                f"方向={order.order_type}, 价格={order.price}, "
                f"委托数量={order.order_volume}, 已成交={order.traded_volume}, "
                f"委托状态={order.order_status}"
            )
            self.outer.order_manager.update_order(order)

        def on_order_stock_async_response(self, response):
            """
            异步下单初始响应回调
            :param response: XtOrderResponse对象
                PS: 仅反馈请求是否被接收，不包含订单实际状态
            """
            if response.error_id == 0:
                self.outer.trade_logger.success(
                    f"异步订单已被柜台受理 | ID: {response.order_id}"
                )
            else:
                self.outer.trade_logger.error(
                    f"异步下单失败 | ID: {response.order_id}"
                )

        def on_cancel_order_stock_async_response(self, response):
            """
            异步撤单初始响应回调
            :param response: XtCancelOrderResponse 对象
            :return:
            """
            if response.error_id == 0:
                self.outer.trade_logger.success(
                    f"异步撤单已被柜台受理 | ID: {response.order_id}"
                )
            else:
                self.outer.trade_logger.error(
                    f"异步撤单失败 | ID: {response.order_id}"
                )
                self.outer.order_manager.handle_cancel_failure(response.order_id)

        def on_stock_trade(self, trade):
            """
            成交回报回调（每成交一笔回调一次）
            :param trade: XtTrade对象
            traded_id	str	成交编号
            traded_time	int	成交时间
            traded_price	float	成交均价
            traded_volume	int	成交数量
            traded_amount	float	成交金额
            order_id	int	订单编号
            """
            order = self.outer.xt_trader.query_stock_order_by_id(trade.order_id)
            self.outer.trade_logger.info(
                f"成交 信息推送: 订单ID={order.order_id}, 股票代码={order.stock_code}, "
                f"方向={order.order_type}, 价格={order.price}, "
                f"委托数量={order.order_volume}, 已成交={order.traded_volume}, "
            )
            if order:
                self.outer.order_manager.update_order(order)

        def on_order_error(self, order_error):
            """
            委托失败回调
            :param order_error:XtOrderError 对象
            """
            self.outer.trade_logger.error(
                "订单委托失败, "
                f"订单编号: {order_error.order_id}, "
                f"下单失败错误码: {order_error.error_id}, "
                f"下单失败具体信息: {order_error.error_msg}"
            )

        def on_cancel_error(self, cancel_error):
            """
            撤单失败回调
            :param cancel_error: XtCancelError 对象
            """
            self.outer.trade_logger.error(
                f"撤单失败 | ID={cancel_error.order_id} | 错误: {cancel_error.error_msg}"
            )
            self.outer.order_manager.handle_cancel_failure(cancel_error.order_id)

    # ---------------------------------------------
    # 定时器
    # ---------------------------------------------
    class RepeatingTimer(threading.Timer):
        """继承Threading.Timer并重写run实现循环定时器"""
        def __init__(self, outer, interval, function):
            super().__init__(interval, function)
            self.outer = outer
            self.interval = interval
            self.daemon = True
            self.name = "timer"

        def run(self):
            while not self.finished.is_set():
                try:
                    self.function()
                except Exception as e_:
                    self.outer.trade_logger.error(f"定时器执行异常: {str(e_)}")
                self.finished.wait(self.interval)


if __name__ == "__main__":
    # 创建交易实例
    TRADE_MINI_QMT_PATH = "D:\国金QMT交易端模拟\\userdata_mini"
    ACCOUNT_ID = "40069146"
    trader = MiniQMTTrader(
        TRADE_MINI_QMT_PATH,
        ACCOUNT_ID,
        session_id=100,
        timer_interval=60
    )

    try:
        # 启动交易系统（非阻塞）
        trader.run()
        # 主线程阻塞直到交易结束
        trader.block_until_end()
    except Exception as e:
        trader.trade_logger.error(f"系统异常: {str(e)}")
    finally:
        # 确保资源释放
        trader.stop()

    # 行情结束，查询当日总体情况
    trader.trade_logger.info("交易结束，查询当日总体情况")
    asset = trader.get_asset()
    orders = trader.get_orders()
    trades = trader.get_trades()
    position = trader.get_positions()
    trader.trade_logger.info(f"当日资产: {asset.total_asset}")
    trader.trade_logger.info(f"当日委托: {len(orders)}笔")
    trader.trade_logger.info(f"当日成交: {len(trades)}笔")
    trader.trade_logger.info(f"持仓数量: {len(position)}只股票")
