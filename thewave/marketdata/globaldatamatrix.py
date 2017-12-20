from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sqlite3
from datetime import date

# from thewave.marketdata.tickerlist import TickerList
import numpy as np
import pandas as pd

from thewave.constants import *
from thewave.marketdata.quandlrequest import QuandlRequest
from thewave.tools.data import panel_fillna


## From Quandl
# ticker, date, open, high, low, close, volume, ex-dividend, plit_ratio,
# .... adj_open, adj_high, adj_low, adj_close, adj_volume

class HistoryManager:
    # if offline ,the ticker_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self, tickers = ('AAPL'), online=True):
        self.initialize_db()

        self._quandl_request = QuandlRequest()

        if online:
            for ticker in tickers:
                print('Online request for DB :',ticker)
                self.update_ticker(ticker)

        # self.__storage_period = FIVE_MINUTES  # keep this as 300
        #self._ticker_number = ticker_number
        self._online = online
        #if self._online:
            #self._asset_list = AssetList(end, volume_average_days, volume_forward)
        #self.__volume_forward = volume_forward
        #self.__volume_average_days = volume_average_days
        self.__tickers = None

    @property
    def tickers(self):
        return self.__tickers

    def quandl_request(self):
        return self._quandl_request

    def initialize_db(self):
        print(DATABASE_DIR)
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            #cursor.execute('CREATE TABLE IF NOT EXISTS History (ticker varchar(20), date date, open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, ex_dividend FLOAT,adj_open FLOAT, adj_high FLOAT, adj_low FLOAT, adj_close FLOAT, adj_volume FLOAT, PRIMARY KEY (ticker, date));')

            cursor.execute('CREATE TABLE IF NOT EXISTS History (ticker varchar(20), date date,'
                            ' open FLOAT, high FLOAT, low FLOAT,'
                            ' close FLOAT, volume FLOAT, ex_dividend FLOAT,'
                            ' adj_open FLOAT, adj_high FLOAT, adj_low FLOAT, adj_close FLOAT, adj_volume FLOAT, '
                            ' PRIMARY KEY (ticker, date));')
            connection.commit()

    def get_global_data_matrix(self, start, end, tickers=('AAPL','A'), features=('close','open')):
        """
        :return a numpy ndarray whose axis is [feature, ticker, time]
        """
        return self.get_global_panel(start, end, tickers, features).values

    # @TODO move to test
    def test_panel(self):
        start = date(2001,1, 1)
        end = date (2017, 12, 1)
        self.get_global_panel(start, end)
        return


    def get_global_panel(self, start, end, tickers=('AAPL','A'),features=('close','open'), online=True):
        """
        :param start/end: linux timestamp in seconds
        :param period: time interval of each data access point default=1 day
        :param features: tuple or list of the feature names
        :return a panel, [feature, ticker, time]
        """

        self.__tickers = tickers
        #@TODO correct order for start and end
        if online:
            for ticker in tickers:
                self.update_ticker(ticker)


        logging.info("feature type list is %s" % str(features))

        connection = sqlite3.connect(DATABASE_DIR)

        # select the min dates
        sqlmin = 'select ticker, min(date) as date from History WHERE ticker IN {ticker} group by ticker'.format(ticker=tickers)
        min_date_frame = pd.read_sql_query(sqlmin, connection, parse_dates=['date'])
        min_date = min_date_frame['date'].max()

        min_date = max (min_date, pd.Timestamp(start))

        #print(' min_date ', min_date)


        #time_index = pd.to_datetime(list(range(int(start), int(end)+1, 3600*24)),unit='s')
        time_index = pd.date_range(start=min_date, end=end,freq='D')
        panel = pd.Panel(items=features, major_axis=tickers, minor_axis=time_index, dtype=np.float32)

        #sql = "SELECT * from History WHERE date > \"{mindate}\" ORDER BY date, ticker".format(mindate=min_date)
        sql = ("SELECT {features}, date, ticker from History WHERE ticker IN {ticker} AND date > \"{mindate}\" "
               "GROUP BY ticker, date".format(features=', '.join(features), ticker=tickers, mindate=min_date))
        #print('sql :', sql)

        df = pd.read_sql_query(sql, connection, parse_dates=['date'], index_col=['date', 'ticker'])

        #print("df dtypes :", df.dtypes)

        #print("DATABASE collected :", df)



        try:
            for ticker in tickers:
                print('DB REQUEST for ticker :', ticker)
                for feature in features:
                    print('feature :', feature)
                    # NOTE: transform the start date to end date
                    if feature == "open":
                        sql = ("SELECT date, adj_open as open FROM History WHERE"
                               " date>=\"{start}\" and date<=\"{end}\"" 
                               " and ticker=\"{ticker}\"".format(
                               start=min_date, end=end, ticker=ticker))
                    elif feature == "high":
                        sql = ("SELECT date, adj_high as high FROM History WHERE"
                               " date>=\"{start}\" and date<=\"{end}\"" 
                               " and ticker=\"{ticker}\"".format(
                               start=min_date, end=end, ticker=ticker))
                    elif feature == "low":
                        sql = ("SELECT date, adj_low as low FROM History WHERE"
                               " date>=\"{start}\" and date<=\"{end}\""
                               " and ticker=\"{ticker}\"".format(
                            start=min_date, end=end, ticker=ticker))
                    elif feature == "close":
                        sql = ("SELECT date, adj_close as close FROM History WHERE"
                               " date>=\"{start}\" and date<=\"{end}\""
                               " and ticker=\"{ticker}\"".format(
                            start=min_date, end=end, ticker=ticker))
                    elif feature == "volume":
                        sql = ("SELECT date, adj_volume as volume FROM History WHERE"
                               " date>=\"{start}\" and date<=\"{end}\""
                               " and ticker=\"{ticker}\"".format(
                            start=min_date, end=end, ticker=ticker))
                    else:
                        msg = ("The feature %s is not supported" % feature)
                        logging.error(msg)
                        raise ValueError(msg)
                    serial_data = pd.read_sql_query(sql, con=connection,
                                                    parse_dates=["date"],
                                                    index_col="date")
                    # serial_data = pd.TimeSeries(serial_data.squeeze(), index = serial_data.index)
                    panel.loc[feature, ticker, serial_data.index] = serial_data.squeeze()
                    panel = panel_fillna(panel, "both")
        finally:
            connection.commit()
            connection.close()
        # return df
        return panel

    # add new history data into the marketdata
    def update_ticker(self, ticker):
        connection = sqlite3.connect(DATABASE_DIR)
        print('update ticker:', ticker)
        try:
            cursor = connection.cursor()
            min_date = cursor.execute('SELECT MIN(date) FROM History WHERE ticker=?;', (ticker,)).fetchall()[0][0]
            max_date = cursor.execute('SELECT MAX(date) FROM History WHERE ticker=?;', (ticker,)).fetchall()[0][0]

            print('min date:', min_date, type(min_date))
            print('max date:', max_date, type(max_date))
            if min_date==None or max_date==None:
                self.fill_ticker(ticker, cursor)
            else:
                #@TODO don't request is up-to-date by checking max_date
                ticker_data = self._quandl_request.data(ticker, {'start_date': max_date})
                for tick in ticker_data:
                    cursor.execute('INSERT OR IGNORE INTO History (date, ticker, open, high, low, close, '
                                   'volume, ex_dividend, '
                                   'adj_open, adj_high, adj_low, adj_close, adj_volume) '
                                   'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (tick[0], ticker, tick[2], tick[3], tick[4], tick[5],
                                    tick[6], tick[7],
                                    tick[8], tick[9], tick[10], tick[11], tick[12]))

            # if there is no data
        finally:
            connection.commit()
            connection.close()

    def fill_ticker(self, ticker, cursor):
        ticker_data = self._quandl_request.data(ticker)
        for tick in ticker_data:
            cursor.execute('INSERT OR IGNORE INTO History (date, ticker, open, high, low, close, '
                           'volume, ex_dividend, '
                           'adj_open, adj_high, adj_low, adj_close, adj_volume) '
                           'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                           (tick[0], ticker, tick[2], tick[3], tick[4], tick[5],
                            tick[6], tick[7],
                            tick[8], tick[9], tick[10], tick[11], tick[12]))

