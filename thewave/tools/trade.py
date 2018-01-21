from __future__ import division,absolute_import,print_function
import numpy as np
from thewave.marketdata.datamatrices import DataMatrices
from thewave.marketdata.globaldatamatrix import HistoryManager
from thewave.tools.configprocess import parse_time
from thewave.constants import *
from thewave.tools.data import get_volume_forward
from time import time


def get_ticker_name_list(config, online):
    """
    :param online: boolean value to show if connected to internet,
    if False, load data from database.
    :return : list of ticker names
    """
    input_config = config["input"]
    if not online:
        start = parse_time(input_config["start_date"])
        end = parse_time(input_config["end_date"])
        volume_forward = get_volume_forward(end - start,
                                            input_config["validation_portion"],
                                            input_config["portion_reversed"])
    else:
        end = time()
        volume_forward = 0
    end = end - (end % input_config["trade_period"])
    start = end - volume_forward - input_config["volume_average_days"] * DAY
    end = end - volume_forward
    tickers = HistoryManager(input_config["ticker_number"], end,
                           volume_forward=volume_forward,
                           volume_average_days=input_config["volume_average_days"],
                           online=online).\
        select_tickers(start, end)
    return tickers


def calculate_pv_after_commission(w1, w0, commission_rate):
    """
    @:param w1: target portfolio vector, first element is btc
    @:param w0: rebalanced last period portfolio vector, first element is btc
    @:param commission_rate: rate of commission fee, proportional to the transaction cost
    """
    mu0 = 1
    mu1 = 1 - 2*commission_rate + commission_rate ** 2
    while abs(mu1-mu0) > 1e-10:
        mu0 = mu1
        mu1 = (1 - commission_rate * w0[0] -
            (2 * commission_rate - commission_rate ** 2) *
            np.sum(np.maximum(w0[1:] - mu1*w1[1:], 0))) / \
            (1 - commission_rate * w1[0])
    return mu1


def get_validation_data(config):
    """
    :return : a 2d numpy array with shape(ticker_number, periods),
     each element the relative price
    """
    config["input"]["feature_list"] = "['close']"
    config["input"]["norm_method"] = "relative"
    config["input"]["global_period"] = config["input"]["global_period"]
    price_matrix = DataMatrices.create_from_config(config)
    validation_set = price_matrix.get_validation_set()["y"][:, 0, :].T
    validation_set = np.concatenate((np.ones((1, validation_set.shape[1])), validation_set), axis=0)
    return validation_set


def get_test_data(config):
    """
    :return : a 2d numpy array with shape(ticker_number, periods),
     each element the relative price
    """
    config["input"]["feature_list"] = "['close']"
    config["input"]["norm_method"] = "relative"
    config["input"]["global_period"] = config["input"]["global_period"]
    price_matrix = DataMatrices.create_from_config(config)
    test_set = price_matrix.get_test_set()["y"][:, 0, :].T
    test_set = np.concatenate((np.ones((1, test_set.shape[1])), test_set), axis=0)
    return test_set

def asset_vector_to_dict(ticker_list, vector, with_cash=True):
    vector = np.squeeze(vector)
    dict_ticker = {}
    if with_cash:
        dict_ticker['cash'] = vector[0]
    for i, name in enumerate(ticker_list):
        if vector[i+1] > 0:
            dict_ticker[name] = vector[i + 1]
    return dict_ticker


def save_validation_data(config, file_name="validation_data", output_format="csv"):
    if output_format == "csv":
        matrix = get_validation_data(config)
        with open(file_name+"."+output_format, 'wb') as f:
            np.savetxt(f, matrix.T, delimiter=";")

