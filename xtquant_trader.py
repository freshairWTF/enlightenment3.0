"""迅投运行脚本"""

from trader import MiniQMTTrader


###############################################################
if __name__ == "__main__":
    # 创建交易实例
    TRADE_MINI_QMT_PATH = "D:\国金QMT交易端模拟\\userdata_mini"
    ACCOUNT_ID = "40069146"
    trader = MiniQMTTrader(
        TRADE_MINI_QMT_PATH,
        ACCOUNT_ID,
        session_id=100,
        timer_interval=10,
        heartbeat_interval=300
    )

    try:
        # 启动交易系统（非阻塞）
        trader.run()
        # 主线程阻塞直到交易结束
        trader.block_until_end()
    except Exception as e:
        trader.trade_logger.error(f"系统异常: {str(e)}")
    finally:
        # 行情结束，查询当日总体情况
        trader.trade_logger.info("交易结束，查询当日总体情况")
        asset = trader.get_asset()
        trader.trade_logger.info(f"当日资产: {asset.total_asset}")
        orders = trader.get_orders()
        trader.trade_logger.info(f"当日委托: {len(orders)}笔")
        trades = trader.get_trades()
        trader.trade_logger.info(f"当日成交: {len(trades)}笔")
        position = trader.get_positions()
        trader.trade_logger.info(f"持仓数量: {len(position)}只股票")
        # 确保资源释放
        trader.stop()
