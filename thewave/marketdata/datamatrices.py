from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import thewave.marketdata.globaldatamatrix as gdm
import numpy as np
import pandas as pd
import logging
import json
from thewave.tools.configprocess import parse_time, parse_list
from thewave.tools.data import get_ticker_list, get_type_list
import thewave.marketdata.replaybuffer as rb

MIN_NUM_PERIOD = 3


class DataMatrices:
    def __init__(self, start, end,
                 start_test, end_test,
                 # period,
                 batch_size=50,
                 # volume_average_days=30,
                 buffer_bias_ratio=0,
                 # market="poloniex",
                 # ticker_filter=1,
                 window_size=50,
                 feature_list=None,
                 validation_portion=0.15,
                 asset_number=2,
                 ticker_list=None,
                 portion_reversed=False,
                 online=False, is_permed=False):
        """
        :type start_test: object
        :param start: Unix time
        :param end: Unix time
        :param access_period: the data access period of the input matrix.
        :param trade_period: the trading period of the agent.
        :param global_period: the data access period of the global price matrix.
                              if it is not equal to the access period, there will be inserted observations
        :param ticker_filter: number of tickers that would be selected
        :param window_size: periods of input data
        :param train_portion: portion of training set
        :param is_permed: if False, the sample inside a mini-batch is in order
        :param validation_portion: portion of cross-validation set
        :param portion_reversed: if False, the order to sets are [train, validation, test]
        else the order is [test, validation, train]
        """

        # assert window_size >= MIN_NUM_PERIOD

        #tickers = get_ticker_list(asset_number)

        tickers = ticker_list
        self.tickers = tickers


        type_list = parse_list(feature_list)
        self.__features = type_list
        self.feature_number = len(type_list)
        self.__history_manager = gdm.HistoryManager(tickers=tickers, online=online)
        # Panel with
        # Items are the features
        # Major_axis for the tickers
        # Minor_axis for the time
        self.__train_validation_data = self.__history_manager.get_global_panel(start,
                                                                     end, tickers=tickers,
                                                                     features=type_list, online= online)

        self.__test_data = self.__history_manager.get_global_panel(start_test,
                                                                     end_test, tickers=tickers,
                                                                     features=type_list, online=online)

        self.__global_data = pd.concat([self.__train_validation_data, self.__test_data], axis=2)
        # ADD Cash
        # tickers = (cash,) + tickers
        self.__ticker_no = len(tickers)

        # add the cash as the first value
        # gd = self.__global_data
        # ones = np.ones((gd.shape[0], gd.shape[2]))
        # df = pd.DataFrame(ones, index = type_list)
        # gd.loc[:,'cash',:]= df

        # portfolio vector memory, [time, assets]
        #@TODO change minor_axis and major_axis
        # major_axis=tickers, minor_axis=time_index
        self.__PVM = pd.DataFrame(index=self.__global_data.minor_axis, columns=self.__global_data.major_axis)
        #self.__PVM = self.__global_data.copy(deep=True)
        self.__PVM = self.__PVM.fillna(1.0 / self.__ticker_no)

        self._window_size = window_size
        #@TODO change minor_axis to timeindex
        # build the period for train + validation + test
        # self._num_periods = self.__global_data.index.levels[0].size
        self._num_train_validation_periods = len(self.__train_validation_data.minor_axis)
        self._num_test_periods = len(self.__test_data.minor_axis)

        self.__divide_data(validation_portion, portion_reversed)

        self._portion_reversed = portion_reversed
        self.__is_permed = is_permed

        self.__batch_size = batch_size
        self.__replay_buffer = None
        self.__delta = 0  # the count of global increased
        # @TODO last index with training+validation+test
        end_index = self._train_ind[-1]
        self.__replay_buffer = rb.ReplayBuffer(start_index=self._train_ind[0],
                                               end_index=end_index,
                                               sample_bias=buffer_bias_ratio,
                                               batch_size=self.__batch_size,
                                               ticker_number=self.__ticker_no,
                                               is_permed=self.__is_permed)

        logging.info("the number of training examples is %s"
                     ", of validation examples is %s" % (self._num_train_samples, self._num_validation_samples))
        logging.debug("the training set is from %s to %s" % (min(self._train_ind), max(self._train_ind)))
        logging.debug("the validation set is from %s to %s" % (min(self._validation_ind), max(self._validation_ind)))
        logging.debug("the window_size is set to %s" % (self._window_size))

    @property
    def global_weights(self):
        return self.__PVM


    @staticmethod
    def create_from_config(config):
        """main method to create the DataMatrices in this project
        @:param config: config dictionary
        @:return: a DataMatrices object
        """
        config = config.copy()
        input_config = config["input"]
        train_config = config["training"]
        backtest_config = config["backtest"]
        start = parse_time(input_config["start_date"])
        end = parse_time(input_config["end_date"])

        # test dates
        start_test = parse_time(backtest_config["start_test_date"])
        end_test = parse_time(backtest_config["end_test_date"])

        # special treatment for lists
        tickers_s = input_config["ticker_list"]
        ticker_list = parse_list(tickers_s)
        asset_number = len(ticker_list)

        features_s = input_config["feature_list"]
        feature_list = parse_list(features_s)

        return DataMatrices(start=start,
                            end=end,
                            start_test=start_test,
                            end_test=end_test,
                            #market=input_config["market"],
                            feature_list=feature_list,
                            window_size=input_config["window_size"],
                            online=input_config["online"],
                            #period=input_config["global_period"],
                            #ticker_filter=input_config["ticker_number"],
                            is_permed=input_config["is_permed"],
                            buffer_bias_ratio=train_config["buffer_biased"],
                            batch_size=train_config["batch_size"],
                            #volume_average_days=input_config["volume_average_days"],
                            validation_portion=input_config["validation_portion"],
                            portion_reversed=input_config["portion_reversed"],
                            asset_number=asset_number,
                            ticker_list=ticker_list,
                            )

    @property
    def global_matrix(self):
        return self.__global_data

    @property
    def train_validation_matrix(self):
        return self.__train_validation_data

    @property
    def test_matrix(self):
        return self.__test_data

    @property
    def ticker_list(self):
        return self.__history_manager.tickers

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @property
    def num_validation_samples(self):
        return self._num_validation_samples

    @property
    def num_test_samples(self):
        return self._num_test_samples

    @property
    def train_indices(self):
        #return self._train_ind[:-(self._window_size + 1):]
        return self._train_ind

    @property
    def validation_indices(self):
        #return self._validation_ind[:-(self._window_size+1):]
        return self._validation_ind

    @property
    def test_indices(self):
        # return self._validation_ind[:-(self._window_size+1):]
        return self._test_ind

    @property
    def test_indices(self):
        #return self._test_ind[:-(self._window_size + 1):]
        return self._test_ind


    def append_experience(self, online_w=None):
        """
        store in the ReplayBuffer a new index
        :param online_w: (number of assets + 1, ) numpy array
        Let it be None if in the backtest case.
        """
        self.__delta += 1
        # @TODO use test_ind
        self._train_ind.append(self._train_ind[-1]+1)
        appended_index = self._train_ind[-1]
        self.__replay_buffer.append_experience(appended_index)

    def get_validation_set(self):
        return self.__pack_samples(self.validation_indices)

    def get_training_set(self):
        #return self.__pack_samples(self._train_ind[:-self._window_size])
        return self.__pack_samples(self.train_indices)

    # @TODO change to REAL test
    def get_test_set(self):
        return self.__pack_samples(self.test_indices)

    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.__pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs-1, :]

        def setw(w):
            self.__PVM.iloc[indexs, :] = w
        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]

        history_close = self.__global_data.iloc[0,:,slice(indexs[0]+self._window_size, indexs[-1]+self._window_size+1 )]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw, "history_close": history_close}

    # volume in y is the volume in next access period
    def get_no_norm_submatrix(self, ind):
        # xs(slice('2017-12-01','2017-12-12'), level='date')
        return self.__global_data.values[:, :, ind:ind+self._window_size+1]
        # return self.__global_data.iloc(axis=0)[ind:ind+self._window_size+1,:]

    # volume in y is the volume in next access period
    def get_submatrix(self, ind):
        # xs(slice('2017-12-01','2017-12-12'), level='date')

        # we extract an array of the last values of 'close' for all tickers and transform to an array
        last = np.array(self.__global_data.loc['close',:,:].iloc[:,ind+self._window_size])
        last = np.reshape(last, (1, len(self.tickers), 1))

        # we recover the sequence including the value after the squence window
        sequence = self.__global_data.values[:,:, ind:ind+self._window_size+1]

        # adjust the values with normalization
        return sequence/last
        # return self.__global_data.iloc(axis=0)[ind:ind+self._window_size+1,:]

    def __divide_data(self, validation_portion, portion_reversed):
        """
        split the data into training, validation and test
        :param validation_portion: part of validation into the training_validation datas
        :param portion_reversed: true is validation means the share of training, false else
        :return:
        """
        train_portion = 1 - validation_portion
        s = float(train_portion + validation_portion)
        if portion_reversed:
            portions = np.array([validation_portion]) / s
            portion_split = (portions * self._num_train_validation_periods).astype(int)
            indices = np.arange(self._num_train_validation_periods)
            self._validation_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self._num_train_validation_periods).astype(int)
            indices = np.arange(self._num_train_validation_periods)
            self._train_ind, self._validation_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[:-(self._window_size + 1)]
        # NOTE(zhengyao): change the logic here in order to fit both
        # reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)


        self._validation_ind = self._validation_ind[:-(self._window_size + 1)]
        self._validation_ind = list(self._validation_ind)
        self._num_validation_samples = len(self._validation_ind)

        # test indices
        self._test_ind = np.arange(start=self._num_train_validation_periods, stop=self._num_train_validation_periods+self._num_test_periods)
        self._test_ind = self._test_ind[:-(self._window_size + 1)]
        self._test_ind = list(self._test_ind)
        self._num_test_samples = len(self._test_ind)
