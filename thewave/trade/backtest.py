from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from thewave.trade import trader
from thewave.marketdata.datamatrices import DataMatrices
from thewave.marketdata.globaldatamatrix import HistoryManager
import logging
from thewave.tools.trade import calculate_pv_after_commission


class BackTest(trader.Trader):
    def __init__(self, config, net_dir=None, agent=None, agent_type="nn"):
        trader.Trader.__init__(self, 0, config, 0, net_dir,
                               initial_cash=1, agent=agent, agent_type=agent_type)
        if agent_type == "nn":
            data_matrices = self._rolling_trainer.data_matrices
        elif agent_type == "traditional":
            config["input"]["feature_list"] = "['close']"
            data_matrices = DataMatrices.create_from_config(config)
        else:
            raise ValueError()
        self.__test_set = data_matrices.get_test_set()
        self.__test_length = self.__test_set["X"].shape[0]
        self._total_steps = self.__test_length
        self._history_close_data = self.__test_set["history_close"].T
        # extract time index
        self._time_index = self._history_close_data.index
        # add cash
        self._history_close_data.insert(0,'cash', np.ones(self._history_close_data.shape[0]))
        self.__shares_matrix = []
        self.__test_pv = 1.0
        self.__test_pc_vector = []
        self.__test_omega = []
        self.__mu =[]

    @property
    def test_set(self):
        return self.__test_set

    @property
    def time_index(self):
        return self._time_index


    @property
    def history_close_data(self):
        """
        cash added
        :return: pandas dataFrame
        """
        return self._history_close_data

    def positions_history(self, prefix=True):
        """
        convert to a DataFrame with the date index
        :return:
        """
        columns_shares = self.ticker_name_list_with_cash
        if prefix:
            columns_shares = ['SHARES ' + x for x in self.ticker_name_list_with_cash]

        shares = pd.DataFrame(self.__shares_matrix, index=self.time_index, columns= columns_shares)
        return shares

    def buy_sell_history(self, prefix=True):
        # percentages of the portfolio for all assets including cash
        test_history = pd.DataFrame(self.test_omega_vector, columns=self.ticker_name_list_with_cash)

        # composition of SHARES in the portfolio for all assets including cash
        shares_history = self.positions_history(prefix=False)

        buy_sell_history = shares_history.diff(periods=1, axis=0)

        # use the start information
        before = np.zeros(buy_sell_history.shape[1])
        before[0] = self.initial_cash
        buy_sell_history.iloc[0] = shares_history.iloc[0].values

        return buy_sell_history

    def returns_data(self):
        return pd.Series(data=self.__test_pc_vector, index=self._time_index)


    def historical_data(self):
        # only use the Data in Database for Backtest
        history_manager = HistoryManager(tickers=self.ticker_name_list, online=False)
        start = self.time_index[0]
        end = self.time_index[-1]
        historical_data = history_manager.historical_data(start=start, end=end, tickers=self.ticker_name_list)
        return historical_data

    def historical_feature_data(self, feature='open'):
        # only use the Data in Database for Backtest
        history_manager = HistoryManager(tickers=self.ticker_name_list, online=False)
        start = self.time_index[0]
        end = self.time_index[-1]
        historical_data = history_manager.historical_feature_data(feature=feature, start=start, end=end, tickers=self.ticker_name_list)
        return historical_data

    def transactions_history(self, prefix=True):
        """
        use the close price from the day before, if necessary use the y ratio
        we use a multi index to isolate the Buy/sell quantity and join to the price
        :param prefix:
        :return:
        """

        # retrieve all BUY/SELL informations
        # stacked
        buy_sell_history = self.buy_sell_history().drop(columns=['cash']).stack()
        # set the name of indexes
        buy_sell_history.index.names = ['date', 'symbol']

        # retrieve the Price History: we take the Open
        historical_open_data = self.historical_feature_data(feature='open').stack()
        # set the name of indexes
        historical_open_data.index.names = ['date', 'symbol']

        transactions_history = pd.concat([buy_sell_history, historical_open_data], axis=1)
        transactions_history.columns = ['amount', 'price']
        transactions_history.reset_index(level=1)

        return transactions_history

    @property
    def test_pv(self):
        return self.__test_pv

    @property
    def test_pc_vector(self):
        return np.array(self.__test_pc_vector, dtype=np.float32)

    @property
    def test_omega_vector(self):
        """

        :return: all the corresponding weights from the tests
        """
        return np.array(self.__test_omega, dtype=np.float32)

    @property
    def test_mu_vector(self):
        return np.array(self.__mu, dtype=np.float32)

    def finish_trading(self):
        self.__test_pv = self._total_capital

        """
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(self._rolling_trainer.data_matrices.sample_count)),
               self._rolling_trainer.data_matrices.sample_count)
        fig.tight_layout()
        plt.show()
        """

    def _log_trading_info(self, time, omega):
        pass

    def _initialize_data_base(self):
        pass

    def _write_into_database(self):
        pass

    def __get_matrix_X(self):
        return self.__test_set["X"][self._steps]

    def __get_matrix_y(self):
        return self.__test_set["y"][self._steps, 0, :]

    def rolling_train(self, online_sample=None):
        self._rolling_trainer.rolling_train()

    def generate_history_matrix(self):
        inputs = self.__get_matrix_X()
        if self._agent_type == "traditional":
            inputs = np.concatenate([np.ones([1, 1, inputs.shape[2]]), inputs], axis=1)
            inputs = inputs[:, :, 1:] / inputs[:, :, :-1]
        return inputs

    def trade_by_strategy(self, omega):
        logging.info("the step is {}".format(self._steps))
        logging.debug("the raw omega is {}".format(omega))
        future_price = np.concatenate((np.ones(1), self.__get_matrix_y()))
        logging.debug("the future price vector is {}".format(future_price))
        quote_price_close = (self.__test_set["history_close"]).iloc[:,self._steps]
        quote_price_previous = np.concatenate( [np.ones(1),
                                np.divide(quote_price_close.values, self.__get_matrix_y()) ])
        # impact of commission (mu)
        pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        self.__mu.append(pv_after_commission)

        # evaluate the shares of assets
        last_capital = self._total_capital*pv_after_commission
        split_assets = np.multiply(last_capital, omega)
        logging.debug("the split of assets is {}".format(split_assets))
        shares = np.divide(split_assets, quote_price_previous)
        self.__shares_matrix.append(shares)
        logging.debug("the number of assets is {}".format(shares))

        portfolio_change = pv_after_commission * np.dot(omega, future_price)
        self._total_capital *= portfolio_change
        self._last_omega = pv_after_commission * omega * \
                           future_price /\
                           portfolio_change
        logging.debug("the portfolio change this period is : {}".format(portfolio_change))
        self.__test_pc_vector.append(portfolio_change)
        self.__test_omega.append(omega)

