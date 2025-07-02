"""全生命周期订单管理器"""

import threading
from datetime import datetime
from xtquant import xtconstant


##########################################################
class OrderManager:
    """券商订单全生命周期管理器"""

    def __init__(self):
        self.orders = {}
        self.lock = threading.RLock()
        # 状态映射表
        self.STATUS_MAP = {
            xtconstant.ORDER_REPORTED: "REPORTED",          # 已报
            xtconstant.ORDER_PART_SUCC: "PART_FILLED",      # 部成
            xtconstant.ORDER_SUCCEEDED: "FILLED",           # 已成
            xtconstant.ORDER_CANCELED: "CANCELED",          # 撤单
            xtconstant.ORDER_JUNK: "JUNK"                   # 废单
        }

    def update_order(
            self,
            xt_order
    ) -> None:
        """
        更新订单状态（线程安全）
        :param xt_order: XtOrder对象
        """
        with self.lock:
            order_id = xt_order.order_id
            # -1 初始化新订单
            if order_id not in self.orders:
                self.orders[order_id] = {
                    "order_id": order_id,
                    "stock_code": xt_order.stock_code,
                    "direction": xt_order.order_type,
                    "price": xt_order.price,
                    "volume": xt_order.order_volume,
                    "status": self.STATUS_MAP.get(xt_order.order_status, "UNKNOWN"),
                    "traded_price": xt_order.traded_price,
                    "traded_volume": xt_order.traded_volume,
                    "create_time": datetime.now(),
                    "update_time": datetime.now(),
                    "cancel_retry": 0,                          # 撤单重试次数
                    "trade_details": [
                        {
                            "time": datetime.now(),
                            "volume":  xt_order.traded_volume,
                            "price": xt_order.traded_price,
                            "value":  xt_order.traded_volume * xt_order.traded_price
                        }
                    ]
                }
            # -2 更新现有订单
            else:
                order = self.orders[order_id]
                prev_volume = order["traded_volume"]
                new_volume = xt_order.traded_volume
                order.update(
                    status=self.STATUS_MAP.get(xt_order.order_status, order["status"]),
                    traded_volume=xt_order.traded_volume,
                    traded_price=xt_order.traded_price,
                    update_time=datetime.now()
                )
                if new_volume > prev_volume:
                    volume_diff = new_volume - prev_volume
                    increment_value = volume_diff * xt_order.traded_price
                    order["trade_details"].append(
                        {
                            "time": datetime.now(),
                            "volume": volume_diff,
                            "price": xt_order.traded_price,
                            "value": increment_value
                        }
                    )

    def mark_canceling(
            self,
            order_id: int
    ) -> None:
        """
        标记撤单中状态
        :param order_id: 订单编号
        """
        with self.lock:
            if order_id in self.orders:
                self.orders[order_id]["status"] = "CANCELING"

    def handle_cancel_failure(
            self,
            order_id: int
    ) -> None:
        """
        处理撤单失败
        :param order_id: 订单编号
        """
        with self.lock:
            if order_id in self.orders:
                order = self.orders[order_id]
                order["cancel_retry"] += 1
                order["status"] = "FAILED_CANCEL" if order["cancel_retry"] > 3 else "CANCELING"

    def get_timeout_orders(
            self,
            timeout_sec: int = 30
    ) -> list:
        """
        获取超时未成交订单
        :param timeout_sec: 超时间隔
        """
        with self.lock:
            now = datetime.now()
            return [
                order for order in self.orders.values()
                if order["status"] in ["REPORTED", "PART_FILLED"]
                   and (now - order["update_time"]).seconds > timeout_sec
            ]

    def cleanup_finalized(
            self,
            timeout_sec: int = 1800
    ) -> None:
        """
        清理终态订单
        :param timeout_sec: 超时间隔
        """
        with self.lock:
            to_delete = []
            for order_id, order in self.orders.items():
                if order["status"] in ["FILLED", "CANCELED", "JUNK"]:
                    # 30分钟后清理终态订单
                    if (datetime.now() - order["update_time"]).seconds > timeout_sec:
                        to_delete.append(order_id)

            for order_id in to_delete:
                del self.orders[order_id]