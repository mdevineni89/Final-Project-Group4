import pandas as pd
from pandas_datareader import DataReader
import datetime as dt

start = dt.datetime(2021,5,1)
end = dt.datetime(2022,3,30)

btc = DataReader('BTC-USD', 'yahoo', start, end)
btc_close = btc['Adj Close']
dates = list(btc.index)
import pandas as pd
prices = pd.DataFrame({'date': dates, 'btc': btc_close.values})
prices.to_csv('cryptocurrency_prices.csv', index=False)
prices