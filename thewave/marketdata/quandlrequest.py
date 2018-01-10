import json
import time
import sys
import quandl
import logging
from datetime import datetime

if sys.version_info[0] == 3:
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import Request, urlopen
    from urllib import urlencode

minute = 60
hour = minute*60
day = hour*24
week = day*7
month = day*30
year = day*365

# Possible Commands
# we focus on the WIKI database from Quandl
# ["Date", "Open", "High", "Low", "Close", "Volume", "Ex-Dividend", "Split Ratio", "Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]
# ["date", "open", "high", "low", "close", "volume", "ex_dividend", "split_ratio", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]
PUBLIC_COMMANDS = ['returnTicker', 'return24hVolume', 'returnOrderBook', 'returnTradeHistory', 'returnChartData', 'returnCurrencies', 'returnLoanOrders']

class QuandlRequest:
    def __init__(self, APIKey='zpFWg7jpwtBPmzA8sT2Z'):
        self.APIKey = APIKey.encode()
        # self.APIVersion = APIVersion.encode()
        # Conversions
        self.timestamp_str = lambda timestamp=time.time(), format="%Y-%m-%d %H:%M:%S": datetime.fromtimestamp(timestamp).strftime(format)
        self.str_timestamp = lambda datestr=self.timestamp_str(), format="%Y-%m-%d %H:%M:%S": int(time.mktime(time.strptime(datestr, format)))
        self.float_roundPercent = lambda floatN, decimalP=2: str(round(float(floatN) * 100, decimalP))+"%"

        quandl.ApiConfig.api_key = APIKey

#####################
# Time Serie #
# params:
#
# Parameter	    Required Type	Values	Description
# database_code	yes	    string	        Code identifying the database to which the dataset belongs.
# dataset_code	yes	    string	        Code identifying the dataset.
# limit	        no	    int		        Use limit=n to get the first n rows of the dataset. Use limit=1 to get just the latest row.
# column_index	no	    int		        Request a specific column. Column 0 is the date column and is always returned. Data begins at column 1.
# start_date	no	    string	yyyy-mm-dd	Retrieve data rows on and after the specified start date.
# end_date	    no	    string	yyyy-mm-dd	Retrieve data rows up to and including the specified end date.
# order	        no	    string	asc/desc	Return data in ascending or descending order of date. Default is desc.
# collapse	    no	    string	none/daily/weekly/monthly/quarterly/annual	Change the sampling frequency of the returned data. Default is none; i.e., data is returned in its original granularity.
# transform	    no	    string	none/diff/rdiff/rdiff_from/cumul/normalize	Perform elementary calculations on the data prior to downloading. Default is none. Calculation options are described below.
#####################


    def data(self, dataset='AAPL',args={}):
        logging.info('Quandl Query for WIKI/%s with args %s' % (dataset, args))
        # print('Quandl Query for: ','WIKI/'+dataset,' with args ', args)
        return quandl.Dataset('WIKI/' + dataset).data(params=args)


    #####################
    # Main Api Function #
    #####################
    def api(self, command, args={}):
        """
        returns 'False' if invalid command or if no APIKey or Secret is specified (if command is "private")
        returns {"error":"<error message>"} if API error
        """
        if command in PUBLIC_COMMANDS:
            url = 'https://poloniex.com/public?'
            args['command'] = command
            ret = urlopen(Request(url + urlencode(args)))
            return json.loads(ret.read().decode(encoding='UTF-8'))
        else:
            return False
