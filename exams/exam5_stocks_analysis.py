import pandas as pd
import yfinance as yf

class StocksAnalysisSummary:

    _DATA_CSV="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/data/apple.json"
    def __init__(self):
        # Setting this option will print all collumns of a dataframe
        pd.set_option('display.max_columns', None)
        # Setting this option will print all of the data in a feature
        pd.set_option('display.max_colwidth', None)


    def q1_extract_stock_price_w_yahoo(self):
        start_date = "2022-12-01"
        end_date = "2022-12-31"


        tesla_tkr_mgr = yf.Ticker("TSLA")
        tesla_prices_dec_2022=tesla_tkr_mgr.history(start=start_date, end=end_date)
        pass


    def q3_extract_stock_price_w_yahoo(self):
        start_date = "2022-12-01"
        end_date = "2022-12-31"


        gme_tkr_mgr = yf.Ticker("GME")
        gme_prices_dec_2022=gme_tkr_mgr.history(start=start_date, end=end_date)
        pass