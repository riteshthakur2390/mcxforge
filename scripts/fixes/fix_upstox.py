from broker.upstox_broker import UpstoxBroker
import upstox_client

b = UpstoxBroker()
api = upstox_client.MarketQuoteApi(b._client)

for test_key in ["NSE_INDEX|Nifty 50", "NSE_INDEX|India VIX"]:
    resp = api.get_full_market_quote(test_key, "2.0")
    print(f"Request: {test_key} -> Response keys: {list(resp.data.keys()) if resp.data else 'None'}")
