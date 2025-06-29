"""xtquant策略测试"""
import os
import random
import time
import json
import threading
# from queue import Queue
from datetime import datetime

from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttrader import XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant

from constant.path_config import DataPATH


##############################################################
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
        self.timer = self.RepeatingTimer(interval, self.on_timer)
        self.timer.daemon = True

        # 心跳检测相关属性
        self.last_heartbeat = datetime.now()            # 上次心跳时间
        self.heartbeat_interval = 300                   # 心跳检测间隔(秒)

        # 创建行情队列（为今后交易优化做准备）
        # self.data_queue = Queue(maxsize=queue_max_size)
        # self.queue_lock = threading.Lock()
        self.last_data = None
        # 订单属性
        self.target_order_dict = self.generate_orders_from_signals(DataPATH.STRATEGIC_TRADING_BOOK)
        self.target_order_dict = self.add_market_suffix(self.target_order_dict)
        self.order_dict = {}
        self.ORDER_MAX_RETRY = 3                              # 最大重试次数
        self.ORDER_MAX_WAIT_TIME = 60                         # 最大等待时间(秒)

    def generate_orders_from_signals(
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
            raise FileNotFoundError(f"交易信号文件不存在: {signal_file_path}")
        with open(signal_file_path, 'r', encoding='utf-8') as f:
            try:
                signals = json.load(f)
                print(f"成功读取交易信号: {len(signals)}条")
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON文件解析错误: {str(e)}") from e

        # -2 查询当前持仓
        positions = self.get_positions()
        position_dict = {pos.stock_code: pos.volume for pos in positions}
        print(f"当前持仓数量: {len(position_dict)}只股票")

        # -3 生成订单集合
        orders_ = []
        for signal in signals:
            stock_code = signal.get("股票代码")
            target_volume = signal.get("买入股数", 0)

            if not stock_code or target_volume < 0:
                print(f"无效信号: 股票代码={stock_code}, 目标数量={target_volume}")
                continue

            current_volume = position_dict.get(stock_code, 0)

            # 计算需要交易的数量
            volume_diff = target_volume - current_volume

            # 买入信号
            if volume_diff > 0:
                order_type = "buy"
            # 卖出信号
            elif volume_diff < 0:
                order_type = "sell"
                volume_diff = abs(volume_diff)
            # 无需交易
            else:
                continue

            orders_.append({
                "stock_code": stock_code,
                "order_type": order_type,
                "volume": volume_diff,
            })
        return orders_

    def add_market_suffix(self, orders):
        """
        为股票订单添加交易所市场标识后缀
        :param orders: 原始订单列表，每个订单是包含stock_code和order_type的字典
        :return: 添加了正确市场后缀的新订单列表
        """
        processed_orders = []

        for order in orders:
            code = str(order["stock_code"])

            # 根据股票代码开头确定交易所标识
            if code.startswith(('600', '601', '603', '605', '688')):
                suffix = '.SH'  # 上交所标识[1,6,8](@ref)
            elif code.startswith(('000', '001', '002', '003')):
                suffix = '.SZ'  # 深交所主板标识[5,8](@ref)
            elif code.startswith(('300', '301')):
                suffix = '.SZ'  # 深交所创业板标识[7,8](@ref)
            elif code.startswith(('8', '9')):
                suffix = '.BJ'  # 北交所标识[7,8](@ref)
            else:
                suffix = ''  # 未知类型保持原样

            # 创建带后缀的新订单
            processed_orders.append({
                "stock_code": code + suffix,
                "order_type": order["order_type"],
                "volume": order["volume"]
            })

        return processed_orders

    def calculate_limit_prices(self, stock_code: str, last_close: float) -> tuple:
        """
        计算股票的涨跌停价格
        :param stock_code: 股票代码（带后缀，如600519.SH）
        :param last_close: 前一日收盘价
        :return: (涨停价, 跌停价)
        """
        # 根据代码前缀确定涨跌幅比例
        code_prefix = stock_code[:3]  # 取前3位（如688）
        if stock_code.startswith("ST") or stock_code.startswith("*ST"):
            limit_ratio = 0.05  # ST股涨跌幅5%
        elif code_prefix in ("688", "30"):  # 科创板/创业板
            limit_ratio = 0.20
        elif code_prefix in ("60", "00", "30"):  # 主板/中小板（30开头需排除创业板）
            # 创业板以30开头但已在上一条件覆盖，此处仅处理主板
            limit_ratio = 0.10
        elif code_prefix in ("8", "9"):  # 北交所
            limit_ratio = 0.30
        else:
            limit_ratio = 0.10  # 默认10%

        # 计算涨跌停价（四舍五入保留2位小数）
        up_limit = round(last_close * (1 + limit_ratio), 2)
        down_limit = round(last_close * (1 - limit_ratio), 2)
        return up_limit, down_limit

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
        """
        -1 启动定时器
        -2 启动行情监听（阻塞式）
        """
        self.timer.start()
        xtdata.run()

    def on_tick(self, data):
        """
        事件驱动处理方法:
            -1 接受、存储最新行情数据
        """
        self.last_data = {
            stock_code: {
                "lastPrice": item["lastPrice"],
                "lastClose": item["lastClose"]
            } for stock_code, item in data.items()
        }

        # with self.queue_lock:
        #     if self.data_queue.full():
        #         self.data_queue.get()
        #     self.data_queue.put(data)

    def on_timer(self):
        """
        定时器回调方法
            -1 查询策略生成的买卖数据 -> 执行订单买卖（sleep控制多个订单生成频率）
            -2 查询订单回调 -> 根据订单状态做进一步处理
        """
        # ----------------------------------
        # 心跳检测
        # ----------------------------------
        self.heartbeat_detection()

        # ----------------------------------
        # 发动策略订单
        # ----------------------------------
        self.send_target_orders()

        # ----------------------------------
        # 检查并处理订单状态（已报待撤/部成待撤/部成 -> 部撤/撤单 -> 再下单）
        # ----------------------------------
        self.check_and_process_orders()

        self.cleanup_finalized_orders()

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
    # 订单管理
    # ---------------------------------------------
    def send_target_orders(self):
        """
        发送target_order_dict中的策略订单
        步骤：
            -1 检查是否有待发送订单
            -2 获取实时行情数据
            -3 生成限价单价格（考虑涨跌停限制）
            -4 批量发送订单
            -5 转移已发送订单到order_dict
        """
        if not self.target_order_dict or self.last_data is None:
            return

        print(f"开始发送策略订单，待处理订单数: {len(self.target_order_dict)}")

        # 遍历并发送订单
        for order_info in list(self.target_order_dict):
            stock_code = order_info["stock_code"]
            order_type = order_info["order_type"]
            volume = order_info["volume"]

            # 此处简化处理，使用最新行情或默认值
            last_price = self.last_data[stock_code]["lastPrice"]
            last_close = self.last_data[stock_code]["lastClose"]
            limit_up, limit_down = self.calculate_limit_prices(stock_code, last_price)

            # 生成委托价格
            if order_type == "buy":
                price = min(last_price * 1.01, limit_up)
            else:
                price = max(last_price * 0.99, limit_down)

            print(f"生成订单: {stock_code} {order_type} {volume}股 @ {price:.2f} "
                  f"[涨停:{limit_up:.2f} 跌停:{limit_down:.2f}]")

            try:
                self.order_stock(
                    stock_code=stock_code,
                    price=round(price, 2),
                    volume=volume,
                    order_type=order_type,
                    price_type=1  # 限价单
                )
            except Exception as e:
                print(f"订单发送异常: {stock_code} | 错误: {str(e)}")

    def check_and_process_orders(self):
        """
        检查所有订单状态并进行相应处理：
        - 超时未成交订单的撤单
        - 撤单成功后重新下单
        - 已完成订单的清理
        """
        current_time = datetime.now()

        for order_info in list(self.order_dict.values()):
            order_id = order_info["订单编号"]
            status = order_info["委托状态"]

            # 计算订单未更新时间差
            time_diff = (current_time - order_info["更新时间"]).total_seconds()

            # 处理超时未成交的订单
            # -1 订单已报/部成 -2 订单成交超时 -3 低于最大重报次数
            if (
                    status in [xtconstant.ORDER_REPORTED, xtconstant.ORDER_PART_SUCC] and
                    time_diff > self.ORDER_MAX_WAIT_TIME and
                    order_info["重试次数"] < self.ORDER_MAX_RETRY
            ):
                print(f"订单 {order_id} 超时未成交({time_diff:.1f}秒)，尝试撤单并重新下单...")
                self.cancel_and_retry_order(order_info)

            # -2 处理终态订单的重试逻辑
            elif (
                    status in [xtconstant.ORDER_CANCELED, xtconstant.ORDER_PART_CANCEL] and
                    order_info.get("重试状态") == "RETRY_PENDING"
            ):
                # 执行重试下单
                self.retry_order_placement(order_info)

    def cancel_and_retry_order(self, order_info):
        """
        撤单并准备重试下单
        """
        order_id = order_info["订单编号"]

        # 标记为重试中
        order_info["重试状态"] = "RETRY_PENDING"
        order_info["更新时间"] = datetime.now()

        # 发起撤单请求
        cancel_result = self.cancel_order(order_id)

        if cancel_result == 0:
            print(f"撤单请求已发送: 订单 {order_id}")
        else:
            print(f"撤单请求失败: 订单 {order_id}, 错误码: {cancel_result}")
            # 重试次数计数
            order_info["重试次数"] += 1
            order_info["重试状态"] = "CANCEL_FAILED"

    def cleanup_finalized_orders(self):
        """终态订单延迟清理（新增方法）"""
        current_time = datetime.now()
        for order_id, order_info in list(self.order_dict.items()):
            status = order_info["委托状态"]
            time_diff = (current_time - order_info["更新时间"]).total_seconds()

            if (status in [
                xtconstant.ORDER_CANCELED, xtconstant.ORDER_PART_CANCEL,
                xtconstant.ORDER_SUCCEEDED, xtconstant.ORDER_JUNK
            ] and time_diff > 60):
                self.order_dict.pop(order_id)
                if status == xtconstant.ORDER_SUCCEEDED:
                    print(f"订单 {order_id} 已完全成交，从订单列表移除")
                elif status in [xtconstant.ORDER_CANCELED, xtconstant.ORDER_PART_CANCEL]:
                    print(f"订单 {order_id} 已取消，从订单列表移除")
                elif status == xtconstant.ORDER_JUNK:
                    print(f"订单 {order_id} 失败，从订单列表移除")

    def retry_order_placement(self, original_order_info):
        """
        重新下单（在撤单完成后调用）
        """
        # 计算剩余数量
        remaining_quantity = original_order_info["委托数量"] - original_order_info.get("成交数量", 0)

        if remaining_quantity <= 0:
            print("无可重试数量，订单已完全成交")
            return None

        # 创建新订单
        try:
            # 获取最新价格 (实际实现需要从行情数据获取)
            # 这里简化处理：使用原订单价格或最新市价
            quote = self.last_data.get(original_order_info["证券代码"], {})
            last_price = quote.get("lastPrice", original_order_info["委托价格"])
            retry_price = last_price * 1.01 if original_order_info["委托类型"] == xtconstant.STOCK_BUY else last_price * 0.99

            # 重新下单
            new_order_id = self.order_stock(
                stock_code=original_order_info["证券代码"],
                price=retry_price,
                volume=remaining_quantity,
                order_type="buy" if original_order_info["委托类型"] == xtconstant.STOCK_BUY else "sell",
                price_type=original_order_info["价格类型"]
            )

            if new_order_id:
                print(f"重试订单已创建: 新订单ID {new_order_id} (原始订单: {original_order_info['订单编号']})")
                return new_order_id
            else:
                print("重试订单创建失败")
                return None

        except Exception as e:
            print(f"重试下单时出错: {str(e)}")
            return None

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
            print(f"心跳超时({time_diff}秒)，启动连接检查...")
            if not self._check_connection():
                print("连接验证失败，触发断连处理")
                self.callback.on_disconnected()
            # 更新心跳时间戳
            self.last_heartbeat = datetime.now()

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
                self.retry_count = 0  # 重置重连计数器
                return True
            return False
        except Exception as e:
            print(f"连接检查异常: {str(e)}")
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
            print("交易连接断开，尝试重连...")
            for _ in range(3):
                if self.outer.xt_trader.reconnect() == 0:
                    self.outer.xt_trader.subscribe(self.outer.account)
                    return
                time.sleep(5)
            raise ConnectionError("重连失败，请检查网络")

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
            order_info = {
                "订单编号": order.order_id,
                "报单时间": order.order_time,
                "证券代码": order.stock_code,
                "委托类型": order.order_type,
                "委托数量": order.order_volume,
                "委托价格": order.price,
                "委托状态": order.order_status,
                "成交均价": order.traded_price,
                "成交数量": order.traded_volume,
                "价格类型": order.price_type
            }

            # -1 更新订单字典
            if order.order_id in self.outer.order_dict:
                self.outer.order_dict[order.order_id].update(order_info)
            # -2 新订单
            else:
                order_info.update({
                    "创建时间": datetime.now(),
                    "更新时间": datetime.now(),
                    "重试次数": 0,
                    "重试状态": ""
                })
                self.outer.order_dict[order.order_id] = order_info

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
            if trade.order_id in self.outer.order_dict:
                order_info = self.outer.order_dict[trade.order_id]
                order_info["更新时间"] = datetime.now()
                total_value = order_info["成交数量"] * order_info["成交均价"] + trade.traded_volume * trade.traded_price
                order_info["成交数量"] += trade.traded_volume
                order_info["成交均价"] = total_value / order_info["成交数量"]

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
            if cancel_error.order_id in self.outer.order_dict:
                order_info = self.outer.order_dict[cancel_error.order_id]
                order_info["重试状态"] = "CANCEL_FAILED"
                order_info["更新时间"] = datetime.now()

                # 重试次数计数
                if "重试次数" in order_info:
                    order_info["重试次数"] += 1

    # ---------------------------------------------
    # 定时器
    # ---------------------------------------------
    class RepeatingTimer(threading.Timer):
        """继承Threading.Timer并重写run实现循环定时器"""
        def run(self):
            while not self.finished.is_set():
                try:
                    self.function(*self.args, ** self.kwargs)
                except Exception as e:
                    print(f"定时器执行异常: {str(e)}")
                self.finished.wait(self.interval)


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
