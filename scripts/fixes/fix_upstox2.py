from broker.upstox_broker import UpstoxBroker
import upstox_client

b = UpstoxBroker()
api = upstox_client.HistoryApi(b._client)

resp = api.get_historical_candle_data1(
    "NSE_INDEX|Nifty 50",
    "day",
    "2026-04-10",
    "2026-04-01",
    "2.0"
)
print("Historical OK:", len(resp.data.candles) if resp.data and resp.data.candles else 0)
