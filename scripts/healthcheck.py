
import os
import pytz
import datetime
from loguru import logger
from broker.factory import get_broker
from utils.market_calendar import latest_expected_trading_day, is_trading_day
from dotenv import load_dotenv

load_dotenv()
IST = pytz.timezone("Asia/Kolkata")

def health_check():
    now = datetime.datetime.now(IST)
    logger.info(f"Health check at {now.isoformat()}")
    
    # 1. Check Market Calendar
    expected_day = latest_expected_trading_day(now.date())
    is_today_trading = is_trading_day(now.date())
    logger.info(f"Market Calendar: Expected Day={expected_day}, Is Today Trading={is_today_trading}")
    
    # 2. Check Broker
    try:
        broker = get_broker()
        logger.info(f"Broker: {broker.broker_name}")
        
        # Test LTP
        nifty_ltp = broker.get_ltp("NIFTY")
        logger.info(f"NIFTY LTP: {nifty_ltp}")
        
        # Test Historical
        start = (now - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        logger.info(f"Fetching historical from {start} to {end}")
        df = broker.get_historical_data("NIFTY", "5minute", start, end)
        
        if df is not None and not df.empty:
            latest_ts = df.index[-1]
            logger.info(f"Latest Candle TS: {latest_ts}")
            logger.info(f"Latest Candle Close: {df['close'].iloc[-1]}")
            logger.info(f"DF Head:\n{df.head(2)}")
            logger.info(f"DF Tail:\n{df.tail(2)}")
            
            if latest_ts.date() < expected_day:
                logger.warning(f"DATA DELAY: Latest candle is from {latest_ts.date()}, but we expected {expected_day}")
            else:
                logger.success("Data is up to date!")
        else:
            logger.error("No historical data returned from broker!")
            
    except Exception as e:
        logger.exception(f"Health check failed: {e}")

if __name__ == "__main__":
    health_check()
