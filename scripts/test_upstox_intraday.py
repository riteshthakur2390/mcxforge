import os
import pytz
import datetime
from loguru import logger
from broker.factory import get_broker
from dotenv import load_dotenv

load_dotenv()
IST = pytz.timezone("Asia/Kolkata")

def test_intraday():
    try:
        broker = get_broker()
        import upstox_client
        api = upstox_client.HistoryApi(broker._client)
        key = "NSE_INDEX|Nifty 50"
        logger.info(f"Fetching intraday for {key}")
        resp = api.get_intra_day_candle_data(key, "5minute", "2.0")
        candles = resp.data.candles
        if candles:
            logger.info(f"Fetched {len(candles)} candles. Latest: {candles[0]}")
        else:
            logger.warning("No candles")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    test_intraday()
