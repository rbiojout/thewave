from unittest import TestCase
import unittest
from thewave.marketdata.datamatrices import DataMatrices
import datetime


class TestDataMatrices(TestCase):
    def setUp(self):
        start = datetime.date(year=2016,month=01,day=01)
        end = datetime.date(year=2016,month=12,day=31)
        self.data = DataMatrices(start=start,end=end)
        self.fake_data = [[[0]]]

    def test_global_weights(self):
        print(self.data)
        gw = self.data.global_weights
        self.assertEqual(len(self.data.tickers), gw.shape[1],msg="number of columns not correct")
        self.assertEqual(1.0, gw.iloc[0][0] * len(self.data.tickers), msg="weights not correct")

    def test_global_matrix(self):
        gm = self.data.global_matrix
        self.assertEqual((3, 10, 366), gm.shape)
        self.assertEqual(101783, int(gm.iloc[0,0,0] * 1000))
        self.assertEqual(8680, int(gm.iloc[0, 9, 0] * 1000))
        self.fail()

    def test_ticker_list(self):
        self.fail()

    def test_num_train_samples(self):
        self.fail()

    def test_test_indices(self):
        self.fail()

    def test_num_test_samples(self):
        self.fail()

    def test_append_experience(self):
        self.fail()

    def test_get_test_set(self):
        self.fail()

    def test_get_training_set(self):
        self.fail()

    def test_next_batch(self):
        self.fail()

    def test_get_submatrix(self):
        self.fail()

if __name__ == "__main__":
    unittest.main()