"""xtquant回调类（重写）"""
import time

from xtquant import xtconstant
from xtquant.xttrader import XtQuantTraderCallback


##########################################################
class TraderCallback(XtQuantTraderCallback):
    """内部回调类，处理交易相关的事件"""

    def __init__(self, outer_instance):
        super().__init__()
        self.outer = outer_instance

    def on_disconnected(self):
        """连接断开回调（可自动重连，最多3次）"""
        self.outer.trade_logger.warning("交易连接断开，尝试重连...")
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
            xtconstant.ACCOUNT_STATUS_INVALID	-1	无效
            xtconstant.ACCOUNT_STATUS_OK	0	正常
            xtconstant.ACCOUNT_STATUS_WAITING_LOGIN	1	连接中
            xtconstant.ACCOUNT_STATUSING	2	登陆中
            xtconstant.ACCOUNT_STATUS_FAIL	3	失败
            xtconstant.ACCOUNT_STATUS_INITING	4	初始化中
            xtconstant.ACCOUNT_STATUS_CORRECTING	5	数据刷新校正中
            xtconstant.ACCOUNT_STATUS_CLOSED	6	收盘后
            xtconstant.ACCOUNT_STATUS_ASSIS_FAIL	7	穿透副链接断开
            xtconstant.ACCOUNT_STATUS_DISABLEBYSYS	8	系统停用（总线使用-密码错误超限）
            xtconstant.ACCOUNT_STATUS_DISABLEBYUSER	9	用户停用（总线使用）
        """
        self.outer.trade_logger.info(
            f"账户信息推送, "
            f"账户ID: {status.account_id}, "
            f"账户类型: {status.account_type}, "
            f"账户状态: {status.status}, "
        )
        if status.status != 0:
            self.outer.trade_logger.error("账户状态异常")

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

        # ---------------------------------------------------
        # 撤单成功时生成新订单写入pending_order
        # ---------------------------------------------------
        if order.order_status == xtconstant.ORDER_CANCELED:
            # -1 检查已经重复下单的次数
            count = sum(1 for order_ in self.outer.pending_orders if order_["stock_code"] == order.stock_code)
            if count > self.outer.ORDER_MAX_RETRY:
                self.outer.trade_logger.warning(f"超过最大挂撤单次数，不再发单: {count}")
            else:
                # 计算未成交量
                unfilled_volume = order.order_volume - order.traded_volume
                if unfilled_volume > 0:
                    # 构造新订单（保留原方向、股票代码）
                    new_order = {
                        "stock_code": order.stock_code,
                        "order_type": "buy" if order.order_type == xtconstant.STOCK_BUY else "sell",
                        "volume": unfilled_volume,
                        "status": "PENDING",    # 标记为待处理
                        "retry_count": 0        # 重置重试次数
                    }
                    # 线程安全地添加至待处理队列
                    with self.outer.pending_orders_lock:

                        self.outer.pending_orders.append(new_order)
                    self.outer.trade_logger.info(
                        f"撤单成功生成新订单 | {order.stock_code} {unfilled_volume}股"
                    )

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
