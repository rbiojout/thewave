from __future__ import absolute_import
from unittest import TestCase
import unittest
from thewave.marketdata.globaldatamatrix import HistoryManager
from datetime import date
import logging


class TestGlobalDataMatrix(unittest.TestCase):
    def setUp(self):
        self.start = date(2017, 1, 1)
        self.end = date(2018, 1, 1)
        self.features= ['close', 'high', 'low']
        self.tickers = ['AAPL','A']
        self.historymanager = HistoryManager(tickers = self.tickers)
        self.datamatrix = self.historymanager.get_global_panel(start=self.start,end=self.end, features=self.features, tickers = self.tickers)

    def test_count_features(self):
        self.assertEqual(self.datamatrix.shape[0], len(self.features))

    def test_count_tickers(self):
        self.assertEqual(self.datamatrix.shape[1], len(self.tickers))

    def test_count_periods(self):
        days = (self.end-self.start).days
        self.assertEqual(self.datamatrix.shape[2], days + 1)


if __name__ == '__main__':
    unittest.main()
