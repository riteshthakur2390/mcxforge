from broker.upstox_broker import UpstoxBroker
import upstox_client

b = UpstoxBroker()
api = upstox_client.MarketQuoteApi(b._client)

for test_key in ["BSE_INDEX|SENSEX", "BSE_INDEX|BSE SENSEX", "NSE_INDEX|SENSEX"]:
    try:
        resp = api.get_full_market_quote(test_key, "2.0")
        print(f"Request: {test_key} -> Response keys: {list(resp.data.keys()) if resp.data else 'None'}")
    except Exception as e:
        print(f"Request: {test_key} -> Error")
