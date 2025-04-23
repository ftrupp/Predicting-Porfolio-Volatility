"""
# Financial Instruments

This script contains links to obtain ticker strings from Wikipedia. It can be extended to other sources to include different types of instruments as well. However, it is important that the ticker name matches the reference name from the data source, such as yfinance.
#

#### 1) Extract the tables from the URLs  
You need to specify the index of the table you want to extract (the tables appear from top to bottom).
"""
import pandas as pd
import json

class fetch_tickers:
    def __init__(self, symbol: str = "S&P 500", file: str = "links_indices.json") -> list:   
        """
        Downloads the instruments that make up an index.      
        :param symbol: Name of the index (e.g., "Nasdaq")
        :param file: Name of the file where the data will be saved.
        """
        self.symbol = symbol
        self.file = file
        self.df = None  # Empty DataFram to initiate
        self.file_path = f"data/{symbol.replace('/', '_')}_data.pkl"

    def fetch_indices(self):
        urls = {
                "S&P 500": {"link": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "index": 0, "country": "United States"},
                "Dow Jones Industrial Average": {"link": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "index": 1, "country": "United States"},
                "NASDAQ-100": {"link": "https://en.wikipedia.org/wiki/NASDAQ-100", "index": 4, "country": "United States"},
                "FTSE 100": {"link": "https://en.wikipedia.org/wiki/FTSE_100_Index", "index": 4, "country": "United Kingdom"},
                "DAX": {"link": "https://en.wikipedia.org/wiki/DAX", "index": 4, "country": "Germany"},
                "CAC 40": {"link": "https://en.wikipedia.org/wiki/CAC_40", "index": 4, "country": "France"},
                "Hang Seng Index": {"link": "https://en.wikipedia.org/wiki/Hang_Seng_Index", "index": 6, "country": "Hong Kong"},
                "ASX 200": {"link": "https://en.wikipedia.org/wiki/S%26P/ASX_200", "index": 2, "country": "Australia"},
                "S&P/TSX Composite Index": {"link": "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index", "index": 3, "country": "Canada"}
        }
        # Load file
        with open(self.file, "w") as file:
            # Convert to json
            json_dict = json.dumps(urls)
            # Save file
            json.dump(json_dict, file)
        urls_document = json.load(open(self.file, "r")) # the "r" means in reading mode
        urls_dict = json.loads(urls_document)
        # Make sure the index is among the available ones.
        assert self.symbol in list(urls_dict.keys()), f"The value must be one of the following {list(urls_dict.keys())}"
        asset_table = pd.read_html(urls_dict[self.symbol]["link"])
        components = asset_table[urls_dict[self.symbol]["index"]]
        tickers_list = components['Symbol'].tolist() 
        return tickers_list

### Below is the code to run it outside of the class.

# Import libraries
import pandas as pd
import json
urls = {
        "S&P 500": {"link": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "index": 0, "country": "United States"},
        "Dow Jones Industrial Average": {"link": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "index": 1, "country": "United States"},
        "NASDAQ-100": {"link": "https://en.wikipedia.org/wiki/NASDAQ-100", "index": 4, "country": "United States"},
        "FTSE 100": {"link": "https://en.wikipedia.org/wiki/FTSE_100_Index", "index": 4, "country": "United Kingdom"},
        "DAX": {"link": "https://en.wikipedia.org/wiki/DAX", "index": 4, "country": "Germany"},
        "CAC 40": {"link": "https://en.wikipedia.org/wiki/CAC_40", "index": 4, "country": "France"},
        "Hang Seng Index": {"link": "https://en.wikipedia.org/wiki/Hang_Seng_Index", "index": 6, "country": "Hong Kong"},
        "ASX 200": {"link": "https://en.wikipedia.org/wiki/S%26P/ASX_200", "index": 2, "country": "Australia"},
        "S&P/TSX Composite Index": {"link": "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index", "index": 3, "country": "Canada"}
}
#### 2) Save data in a json file.

file_name = "links_indices.json"
with open(file_name, "w") as file:
    # Convert to json
    json_dict = json.dumps(urls)
    # Save file
    json.dump(json_dict, file)

#### 3) Create the function to obtain the list of components of an index.
# By default: S&P 500

def get_assets(ticker: str = "S&P 500", file_name: str = "links_indices.json") -> pd.DataFrame:
    
    """
    Downloads the assets that make up an index.
    """
    
    # Load document
    urls_document = json.load(open(file_name, "r")) # "r" means reading mode
    urls_dict = json.loads(urls_document)
    # Make sure the index is among the available ones.
    assert ticker in list(urls_dict.keys()), f"The value must be one of the following {list(urls_dict.keys())}"
    asset_table = pd.read_html(urls_dict[ticker]["link"])
    components = asset_table[urls_dict[ticker]["index"]]
    
    return components


#### 4) Instantiate the function
# In this case, get the tickers of the 500 companies that make up the S&P 500.  
# Store the tickers in a list and optionally add any additional tickers of interest.  
# Instantiate the function more than once to get data from different indices and combine the lists.

if __name__ == "__main__":
    
    # Definir índice
    ticker = "S&P 500"
    file_name = "links_indices.json"
    
    # Get assets
    fetcher = fetch_tickers(ticker, file_name)
    tickers_list = fetcher.fetch_indices()
    # Additional custom list of tickers
    tickers_list = tickers_list + ["BABA", "MELI", "YPF", "VIST", "GGAL", "PAM", "BMA", "SUPV", "QBTS", "IONQ", "RGTI"]
    print(tickers_list)