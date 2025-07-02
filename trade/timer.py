"""定时器"""
import threading

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
