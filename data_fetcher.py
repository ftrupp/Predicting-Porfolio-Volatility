"""
# Data Fetcher
This script contains the classes that allow importing data from Binance and Yahoo Finance.
1. The parameters are defined: symbol, timeframe, and start_date. For Yahoo, end_date and a filename to save the file are also defined.
2. The class is instantiated.
3. The data is fetched.
4. Then it has to be loaded into a dataframe.

### Binance
from modules.data_fetcher import BinanceDataFetcher  
symbol = 'BTC/USDT'  
timeframe = '1h'  
start_date = '2024-08-01T00:00:00Z'  
fetcher = BinanceDataFetcher(symbol=symbol, timeframe=timeframe, start_date=start_date+'T00:00:00Z')  
fetcher.fetch_data()  
df = fetcher.load_data()  

### Yahoo
from modules.data_fetcher import YahooDataFetcher  
symbol = tickers_lista  
timeframe = "1d"  
start_date = "2015-01-01"  
end_date = "2025-03-09"  
name = 'allstocks'
fetcher = YahooDataFetcher(tickers=symbol, interval=timeframe, start_date=start_date, end_date=end_date, name = name)  
fetcher.fetch_data()  
df = fetcher.load_data()  
"""

import pandas as pd
import time
import os

try:
    import ccxt
except ImportError:
    print("Installing ccxt...")
    os.system('pip install ccxt')
    import ccxt

class BinanceDataFetcher:
    def __init__(self, symbol: str, timeframe: str = '1d', start_date: str = '2020-01-01T00:00:00Z', limit: int = 1000):
        """
        Initializes the object to fetch historical data from Binance.
        :param symbol: Trading pair (e.g., 'BTC/USDT')
        :param timeframe: Candle interval (e.g., '1d', '4h', '1h')
        :param start_date: Start date in ISO8601 format (e.g., '2020-01-01T00:00:00Z')
        :param limit: Maximum number of candles per request
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.limit = limit
        self.exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        self.df = None  # Empty DataFrame 
        self.file_path = f"data/{symbol.replace('/', '_')}_dataBinance.pkl"

    def fetch_data(self):
        """Fetches the historical data and stores it in a DataFrame.."""
        print('Downloading data...')
        since = self.exchange.parse8601(self.start_date)
        all_data = []

        while True:
            # Obtener datos OHLCV
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=self.limit)

            if len(ohlcv) == 0:
                break

            # Convert to DataFrame. Starts with a temporary one:
            df_prov = pd.DataFrame(
                ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            df_prov['Timestamp'] = pd.to_datetime(df_prov['Timestamp'], unit='ms')

            all_data.append(df_prov)

            # Advance the initial timestamp to the last obtained point.
            since = ohlcv[-1][0] + 1  

            # Sleep to avoid exceeding the call rate limit.
            time.sleep(self.exchange.rateLimit / 1000)

        # Merge all the data into a single DataFrame.
        self.df = pd.concat(all_data, ignore_index=True)

        # Clean and convert indices. Change the timezone to Buenos Aires time.
        self.df = self.df.dropna(subset=['Timestamp'])
        self.df.set_index('Timestamp', inplace=True)
        #self.df.index = self.df.index.tz_localize('UTC').tz_convert('America/Argentina/Buenos_Aires')
        self.df.to_pickle(self.file_path)  # Save data

            
    # This would be to directly obtain the dataframe with the data.
    # def get_dataframe(self) -> pd.DataFrame:
    #     """Returns the DataFrame with the historical data."""
    #     if self.df is None:
    #         raise ValueError("No data has been downloaded yet. Run fetch_data() first.")
    #     return self.df

    # Retrieve the data from the saved file.
    def load_data(self):
        if os.path.exists(self.file_path):
            return pd.read_pickle(self.file_path)
        else:
            return self.fetch_data()

### Get data from Yahoo Finance


from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    print("Instalando yfinance...")
    import os
    os.system('pip install yfinance')

# If the code is executed in a .py script, we get the path of the current file.
if "__file__" in globals():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
# If we are in a notebook, we use the working directory path.
    base_path = os.getcwd() 

data_folder = os.path.join(base_path, "data")

# Crear la carpeta si no existe
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

class YahooDataFetcher:
    def __init__(self, tickers: str, interval: str = '1d', start_date: str = "2020-01-01", end_date: str = None, name: str = 'no_name'):
        if end_date is None:
            end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        """
        Initializes the object to fetch historical data from Yahoo Finance.  
        :param tickers: stock symbol (e.g., 'AAPL')  
        :param timeframe: Candle interval (e.g., '1m', '15m', '1h', '4h', '1d', '1wk', '1mo')  
        :param start_date: Start date in "YYYY-MM-DD" format (e.g., "2020-01-01")  
        :param end_date: End date in "YYYY-MM-DD" format (e.g., "2020-01-01"). If not provided, it defaults to yesterday.
        """
        self.tickers = tickers
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.df = None  # Empty DataFrame to initiate
        self.name = name
        self.file_path = os.path.join(data_folder, f"{name}_dataYfinance.pkl")

    def fetch_data(self):
        # It can download multiple tickers at once with threads = True.  
        self.df = yf.download(tickers = self.tickers, start = self.start_date, end = self.end_date, interval = self.interval, auto_adjust=True)
        # To remove the multi-level index (the second row with the ticker):
        #self.df.columns = self.df.columns.droplevel(1)
        self.df.to_pickle(self.file_path)  # Save data

    # Retrieve the data from the saved file.
    def load_data(self):
        if os.path.exists(self.file_path):
            return pd.read_pickle(self.file_path)
        else:
            return self.fetch_data()