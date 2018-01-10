from unittest import TestCase
import unittest
from thewave.marketdata.datamatrices import DataMatrices
import datetime
import numpy as np
import pandas as pd
from datetime import date
import logging


class TestDataMatrices(TestCase):
    def setUp(self):
        self.start = date(2017, 1, 1)
        self.end = date(2018, 1, 1)
        self.start_test = date(2017, 1, 1)
        self.end_test = date(2018, 1, 1)
        self.window_size = 50
        self.validation_portion = 0.15
        self.feature_list= ['close', 'high', 'low']
        self.feature_number = len(self.feature_list)
        self.ticker_list = ['AAPL', 'A']
        self.ticker_number = len(self.ticker_list)
        self.data = DataMatrices(start=self.start,end=self.end,
                                 start_test=self.start_test, end_test=self.end_test,
                                 validation_portion=self.validation_portion,
                                 window_size = self.window_size,
                                 feature_list=self.feature_list,
                                 ticker_list = self.ticker_list)
        self.fake_data = [[[0]]]

    def test_global_weights(self):
        days = (self.end - self.start).days +1 + (self.end-self.start).days + 1
        global_weights = self.data.global_weights

        self.assertTrue(isinstance(global_weights, pd.core.frame.DataFrame))
        self.assertEqual((days,self.ticker_number), global_weights.shape)
        # number of columns not correct
        self.assertEqual(self.ticker_number, global_weights.shape[1])
        # weights not correct
        self.assertEqual(1.0, global_weights.iloc[0][0] * self.ticker_number)


    def test_global_matrix(self):
        days = (self.end - self.start).days + 1 + (self.end - self.start).days + 1
        gm = self.data.global_matrix

        self.assertTrue(isinstance(gm, pd.core.panel.Panel))
        self.assertEqual((self.feature_number, self.ticker_number, days), gm.shape)
        self.assertSequenceEqual(gm.axes[0].values.tolist(), self.feature_list)
        self.assertSequenceEqual(gm.axes[1].values.tolist(), self.ticker_list)

        self.assertTrue(gm.axes[2].values[0], pd.Timestamp(self.start).to_datetime64())
        self.assertTrue(gm.axes[2].values[-1], pd.Timestamp(self.end_test).to_datetime64())

    def test_train_validation_matrix(self):
        days = (self.end - self.start).days + 1
        tvm = self.data.train_validation_matrix

        self.assertTrue(isinstance(tvm, pd.core.panel.Panel))
        self.assertEqual((self.feature_number, self.ticker_number, days), tvm.shape)
        self.assertSequenceEqual(tvm.axes[0].values.tolist(), self.feature_list)
        self.assertSequenceEqual(tvm.axes[1].values.tolist(), self.ticker_list)
        # start is reflected
        self.assertEqual(tvm.axes[2].values[0], pd.Timestamp(self.start).to_datetime64())
        # end is reflected
        self.assertEqual(tvm.axes[2].values[-1], pd.Timestamp(self.end).to_datetime64())

    def test_test_matrix(self):
        days = (self.end - self.start).days + 1
        tm = self.data.test_matrix

        self.assertTrue(isinstance(tm, pd.core.panel.Panel))
        self.assertEqual((self.feature_number, self.ticker_number, days), tm.shape)
        self.assertSequenceEqual(tm.axes[0].values.tolist(), self.feature_list)
        self.assertSequenceEqual(tm.axes[1].values.tolist(), self.ticker_list)
        # start is reflected
        self.assertEqual(tm.axes[2].values[0], pd.Timestamp(self.start_test).to_datetime64())
        # end is reflected
        self.assertEqual(tm.axes[2].values[-1], pd.Timestamp(self.end_test).to_datetime64())

    def test_ticker_list(self):
        self.assertSequenceEqual(self.ticker_list, self.data.ticker_list)

    def test_num_train_samples(self):
        train_portion = 1 - self.validation_portion
        s = float(train_portion + self.validation_portion)
        num_train_validation_periods = len(self.data.train_validation_matrix.minor_axis)
        num_train = int(num_train_validation_periods * (1 - self.validation_portion) / s) - (self.window_size + 1)
        # number of train sample is not well split
        self.assertEqual(num_train, self.data.num_train_samples)

    def test_num_validation_samples(self):
        days = (self.end - self.start).days + 1
        self.assertEqual(days, self.data.num_train_samples + (self.window_size + 1) + self.data.num_validation_samples + (self.window_size + 1) )
        self.assertEqual(days - self.data.num_train_samples - (self.window_size + 1), self.data.num_validation_samples - (self.window_size + 1))

    def test_num_validation_samples(self):
        days = (self.end_test - self.start_test).days + 1
        # , msg="validation size reduced by window size"
        self.assertEqual(days, self.data.num_test_samples + (self.window_size + 1) )


    def test_train_indices(self):
        train_ind = self.data.train_indices
        # start at 0
        self.assertEqual(train_ind[0], 0)
        # last indice for train taking window size
        self.assertEqual(train_ind[-1], self.data.num_train_samples - 1)


    def test_validation_indices(self):
        validation_ind = self.data.validation_indices
        # start after training indices
        self.assertEqual(validation_ind[0], self.data.num_train_samples + (self.window_size + 1) )
        # last indice without window size
        self.assertEqual(validation_ind[-1], self.data.num_validation_samples
                         + self.data.num_train_samples + (self.window_size + 1)
                         - 1)

    def test_test_indices(self):
        test_ind = self.data.test_indices
        # start after training indices and validation
        self.assertEqual(test_ind[0], self.data.num_train_samples + (self.window_size + 1) + self.data.num_validation_samples + (self.window_size + 1))
        # last indice without window size
        self.assertEqual(test_ind[-1], self.data.num_train_samples + (self.window_size + 1)
                         + self.data.num_validation_samples + (self.window_size + 1)
                         + self.data.num_test_samples
                         - 1)

if __name__ == "__main__":
    unittest.main()