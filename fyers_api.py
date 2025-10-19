import datetime as dt
import pandas as pd
from config.trading_params import CLIENT_ID, SECRET_KEY, REDIRECT_URI, AUTH_CODE
from fyers_apiv3 import fyersModel


def initialize_fyers():
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        grant_type="authorization_code",
        redirect_uri=REDIRECT_URI
    )
    session.set_token(AUTH_CODE)
    resp = session.generate_token()
    if "access_token" in resp:
        return fyersModel.FyersModel(
            client_id=CLIENT_ID,
            token=resp["access_token"],
            log_path=""
        )
    else:
        raise RuntimeError(f"Failed to generate access token: {resp}")


def get_historical_data(fyers, symbol, resolution, num_candles):
    now = dt.datetime.now()
    minutes_back = int(resolution) * int(num_candles)
    start_time = now - dt.timedelta(minutes=minutes_back)
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "0",
        "range_from": str(int(start_time.timestamp())),
        "range_to": str(int(now.timestamp())),
        "cont_flag": "1"
    }
    resp = fyers.history(data)
    if isinstance(resp, dict):
        status = resp.get("s")
        candles = resp.get("candles", [])
        if status == "ok" and candles:
            df = pd.DataFrame(candles, columns=[
                'timestamp','open','high','low','close','volume'
            ])
            df['timestamp'] = (pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata'))
            return df
        elif status == "no_data":
            return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
        else:
            raise RuntimeError(f"Error fetching data: {resp}")
    else:
        raise RuntimeError(f"Unexpected response: {resp}")


def place_order(fyers, symbol, side, quantity, order_type, limit_price=None, stop_price=None):
    data = {
        "symbol": symbol,
        "qty": quantity,
        "type": 2 if order_type == 'MARKET' else 1, # 1 for limit, 2 for market
        "side": 1 if side == 'BUY' else -1, # 1 for buy, -1 for sell
        "productType": "INTRADAY",
        "limitPrice": limit_price if limit_price else 0,
        "stopPrice": stop_price if stop_price else 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": "False",
    }
    response = fyers.place_order(data=data)
    return response

def cancel_order(fyers, order_id):
    data = {"id": order_id}
    response = fyers.cancel_order(data=data)
    return response

def get_orders(fyers):
    response = fyers.get_orders()
    return response


def get_historical_range(fyers, symbol, resolution, start_dt, end_dt):
    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "0",
        "range_from": str(int(start_dt.timestamp())),
        "range_to": str(int(end_dt.timestamp())),
        "cont_flag": "1"
    }
    resp = fyers.history(data)
    if isinstance(resp, dict):
        status = resp.get("s")
        candles = resp.get("candles", [])
        if status == "ok" and candles:
            df = pd.DataFrame(candles, columns=[
                'timestamp','open','high','low','close','volume'
            ])
            df['timestamp'] = (pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata'))
            return df
        elif status == "no_data":
            return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
        else:
            raise RuntimeError(f"Error fetching data: {resp}")
    else:
        raise RuntimeError(f"Unexpected response: {resp}")