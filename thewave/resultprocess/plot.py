from __future__ import absolute_import, print_function, division

import pickle

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import pandas as pd
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from thewave.marketdata.globaldatamatrix import HistoryManager
from thewave.tools.indicator import max_drawdown, sharpe, positive_count, negative_count, moving_accumulate
from thewave.tools.configprocess import parse_time, check_input_same
from thewave.tools.shortcut import execute_backtest, get_backtester


# the dictionary of name of indicators mapping to the function of related indicators
# input is portfolio changes
#@TODO adjust regarding period, at this time period of 1 day
INDICATORS = {"portfolio value": np.prod,
              "sharpe ratio": sharpe,
              "max drawdown": max_drawdown,
              "positive periods": positive_count,
              "negative periods": negative_count,
              "postive day": lambda pcs: positive_count(moving_accumulate(pcs, 1)),
              "negative day": lambda pcs: negative_count(moving_accumulate(pcs, 1)),
              "postive week": lambda pcs: positive_count(moving_accumulate(pcs, 7)),
              "negative week": lambda pcs: negative_count(moving_accumulate(pcs, 7)),
              "average": np.mean}

NAMES = {"best": "Best Stock (Benchmark)",
         "crp": "UCRP (Benchmark)",
         "ubah": "UBAH (Benchmark)",
         "anticor": "ANTICOR",
         "olmar": "OLMAR",
         "pamr": "PAMR",
         "cwmr": "CWMR",
         "rmr": "RMR",
         "ons": "ONS",
         "up": "UP",
         "eg": "EG",
         "bk": "BK",
         "corn": "CORN",
         "m0": "M0",
         "wmamr": "WMAMR"
         }

def plot_backtest(config, algos, labels=None):
    """
    @:param config: config dictionary
    @:param algos: list of strings representing the name of algorithms or index of thewave result
    """
    results = []
    for i, algo in enumerate(algos):
        if algo.isdigit():
            #results.append(np.cumprod(_load_from_summary(algo, config)))
            results.append(np.cumprod(execute_backtest(algo, config)))
            logging.info("load index "+algo+" from csv file")
        else:
            logging.info("start executing "+algo)
            results.append(np.cumprod(execute_backtest(algo, config)))
            logging.info("finish executing "+algo)

    start, end = _extract_test(config)
    #timestamps = np.linspace(start, end, len(results[0]))
    timestamps = np.linspace(pd.Timestamp(start).value, pd.Timestamp(end).value, len(results[0]), dtype=np.int64)
    #dates = [start + datetime.timedelta(days=x) for x in range((end-start).days + 1)]
    dates = [start + timedelta(days=x) for x in range(len(results[0]))]

    #dates = [datetime.datetime.fromtimestamp(int(ts)-int(ts)%config["input"]["global_period"])
     #        for ts in timestamps]

    weeks = mdates.WeekdayLocator()
    days = mdates.DayLocator()

    #rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"],
    #              "size": 8})

    """
    styles = [("-", None), ("--", None), ("", "+"), (":", None),
              ("", "o"), ("", "v"), ("", "*")]
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5)
    for i, pvs in enumerate(results):
        if len(labels) > i:
            label = labels[i]
        else:
            label = NAMES[algos[i]]
        #ax.semilogy(dates, pvs, linewidth=1, label=label)
        ax.plot(dates, pvs, linewidth=1, label=label)

    plt.ylabel("portfolio value $p_t/p_0$", fontsize=12)
    plt.xlabel("time", fontsize=12)
    #xfmt = mdates.DateFormatter("%m-%d %H:%M")
    xfmt = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_minor_locator(days)
    datemin = dates[0]
    datemax = dates[-1]
    ax.set_xlim(datemin, datemax)

    ax.xaxis.set_major_formatter(xfmt)
    plt.grid(True)
    plt.tight_layout()
    ax.legend(loc="upper left", prop={"size":10})
    fig.autofmt_xdate()
    #plt.savefig("./train_package/"+algo+"result"+algo+".eps", bbox_inches='tight', pad_inches=0)
    plt.savefig("result.eps", bbox_inches='tight', pad_inches=0)
    plt.show()


def table_backtest(config, algos, labels=None, format="raw",
                   indicators=list(INDICATORS.keys())):
    """
    @:param config: config dictionary
    @:param algos: list of strings representing the name of algorithms
    or index of thewave result
    @:param format: "raw", "html", "latex" or "csv". If it is "csv",
    the result will be save in a csv file. otherwise only print it out
    @:return: a string of html or latex code
    """
    results = []
    labels = list(labels)
    for i, algo in enumerate(algos):
        if algo.isdigit():
            # @TODO use the test and not the validation segment
            #portfolio_changes = _load_from_summary(algo, config)
            portfolio_changes = execute_backtest(algo, config)
            logging.info("load index " + algo + " from csv file")
        else:
            logging.info("start executing " + algo)
            portfolio_changes = execute_backtest(algo, config)
            logging.info("finish executing " + algo)

        indicator_result = {}
        for indicator in indicators:
            indicator_result[indicator] = INDICATORS[indicator](portfolio_changes)
        results.append(indicator_result)
        if len(labels)<=i:
            labels.append(NAMES[algo])

    dataframe = pd.DataFrame(results, index=labels)

    start, end = _extract_validation(config)
    #start = datetime.datetime.fromtimestamp(start - start%config["input"]["global_period"])
    #end = datetime.datetime.fromtimestamp(end - end%config["input"]["global_period"])

    print("backtest start from "+ str(start) + " to " + str(end))
    if format == "html":
        print(dataframe.to_html())
    elif format == "latex":
        print(dataframe.to_latex())
    elif format == "raw":
        print(dataframe.to_string())
    elif format == "csv":
        dataframe.to_csv("./compare"+end.strftime("%Y-%m-%d")+".csv")
    else:
        raise ValueError("The format " + format + " is not supported")


def file_backtest(config, algo):
    logging.info("start executing back test")
    backtest = get_backtester(algo, config)
    # print("test omega :", backtest.test_omega_vector)

    ticker_name_list_no_cash = backtest.ticker_name_list
    ticker_name_list_with_cash = backtest.ticker_name_list_with_cash

    # closing prices
    stock_history = backtest.history_close_data
    # composition of SHARES in the portfolio
    positions_history = backtest.positions_history()

    # stock history
    # need for the date time index to be present
    #lastclose = pd.DataFrame(backtest.validation_set['lastclose'])
    #lastclose.columns = tickers
    #lastclose['cash'] = np.ones(lastclose.shape[0])

    # extract the returns as a pandas Series (Daily returns of the strategy, noncumulative)
    returns = backtest.returns_data()

    # extract the positions as a pandas Dataframe
    # Daily net position values.
    #   - Time series of dollar amount invested in each position and cash.
    #   - Days where stocks are not held can be represented by 0 or NaN.
    #   - Non - working capital is labelled 'cash'
    #   - Example:
    #       index           'AAPL'         'MSFT'           cash
    #       2004 - 01 - 09  13939.3800      -14012.9930     711.5585
    #       2004 - 01 - 12  14492.6300      -14624.8700     27.1821
    #       2004 - 01 - 13  -13853.2800     13653.6400      -43.6375
    positions = backtest.positions_history(prefix=False)

    # extract the transactions as a pandas DataFrame
    # Executed trade volumes and fill prices.
    #   - One row per trade.
    #   - Trades on different names that occur at the same time will have identical indicies.
    #   - Example:
    #       index                       amount      price       symbol
    #       2004 - 01 - 09  12:18:01    483         324.12      'AAPL'
    #       2004 - 01 - 09  12:18:01    122         83.10       'MSFT'
    #       2004 - 01 - 13  14:12:23    -75         340.43      'AAPL'
    transactions = backtest.transactions_history()

    historical_data = backtest.historical_data()

    # prices per step
    print("test pv :", backtest.test_pc_vector)
    # tickers.insert(0,'cash')
    label_history = ['Omega '+ x for x in ticker_name_list_with_cash]
    test_history = pd.DataFrame(backtest.test_omega_vector, columns=label_history)
    test_history = test_history.set_index(stock_history.index.values)
    test_history['pv']=np.cumprod(backtest.test_pc_vector)
    test_history['mu'] = backtest.test_mu_vector

    buy_sell_history = backtest.buy_sell_history(prefix=True)

    # remove utc reference for joining and convert to naive datetime
    stock_history.index = stock_history.index.tz_localize(None)
    positions_history.index = positions_history.index.tz_localize(None)
    test_history.index = test_history.index.tz_localize(None)
    buy_sell_history.index = buy_sell_history.index.tz_localize(None)

    frames = [stock_history, positions_history, test_history, buy_sell_history]
    result = pd.concat(frames, axis=1)

    with open("./train_package/"+algo+"/backtest-"+algo+".pickle", "wb") as f:
        pickle.dump([returns, positions, transactions], f)

    result.to_csv("./train_package/"+algo+"/backtest-"+algo+".csv", mode='w')
    writer = pd.ExcelWriter("./train_package/"+algo+"/backtest-"+algo+".xlsx")
    result.to_excel(excel_writer=writer, sheet_name='Backtest')
    historical_data.to_excel(excel_writer=writer, sheet_name='Historical')
    returns.to_excel(excel_writer=writer, sheet_name='Returns')
    positions.to_excel(excel_writer=writer, sheet_name='positions')
    transactions.to_excel(excel_writer=writer, sheet_name='transactions')

    writer.save()
    logging.info("finish executing back test")

    print("THIS IS THE END")


def _extract_validation(config):
    global_start = parse_time(config["input"]["start_date"])
    global_end = parse_time(config["input"]["end_date"])
    span = global_end - global_start
    start = global_end - timedelta(int(config["input"]["validation_portion"] * span.days))
    end = global_end
    return start, end

def _extract_test(config):
    start = parse_time(config["backtest"]["start_test_date"])
    window_size = int(config["input"]["window_size"])
    start = start + timedelta(days = window_size +1 )
    end = parse_time(config["backtest"]["end_test_date"])
    return start, end

def _load_from_summary(index, config):
    """ load the backtest result form train_package/train_summary
    @:param index: index of the training and backtest
    @:return: numpy array of the portfolio changes
    """
    dataframe = pd.DataFrame.from_csv("./train_package/train_summary.csv")
    histoframe = dataframe.loc[int(index)]
    # take the last one in case of multiple reruns
    if isinstance(histoframe, pd.core.frame.DataFrame):
        histoframe = histoframe.iloc[-1, :]

    history_string = histoframe["backtest_test_history"]
    if not check_input_same(config, json.loads(histoframe["config"])):
        raise ValueError("the date of this index is not the same as the default config")
    return np.fromstring(history_string, sep=",")[:-1]

def _evaluate_test(index, config):
    """     """
    dataframe = pd.DataFrame.from_csv("./train_package/train_summary.csv")
    histoframe = dataframe.loc[int(index)]
    # take the last one in case of multiple reruns
    if isinstance(histoframe, pd.core.frame.DataFrame):
        histoframe = histoframe.iloc[-1, :]

    history_string = histoframe["backtest_test_history"]
    if not check_input_same(config, json.loads(histoframe["config"])):
        raise ValueError("the date of this index is not the same as the default config")
    return np.fromstring(history_string, sep=",")[:-1]
