from __future__ import absolute_import, print_function, division
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import pandas as pd
import logging
import json
import numpy as np
import datetime
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
            results.append(np.cumprod(_load_from_summary(algo, config)))
            logging.info("load index "+algo+" from csv file")
        else:
            logging.info("start executing "+algo)
            results.append(np.cumprod(execute_backtest(algo, config)))
            logging.info("finish executing "+algo)

    start, end = _extract_test(config)
    #timestamps = np.linspace(start, end, len(results[0]))
    timestamps = np.linspace(pd.Timestamp(start).value, pd.Timestamp(end).value, len(results[0]), dtype=np.int64)
    #dates = [start + datetime.timedelta(days=x) for x in range((end-start).days + 1)]
    dates = [start + datetime.timedelta(days=x) for x in range(len(results[0]))]

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
            portfolio_changes = _load_from_summary(algo, config)
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

    start, end = _extract_test(config)
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
    print("test omega :", backtest.test_omega_vector)
    start, end = _extract_test(config)
    tickers = backtest._ticker_name_list
    history = HistoryManager(tickers=tickers,online=False)
    datas = history.get_global_panel(start=start,end=end,tickers=tickers,features=['close'],online=False)

    window_size = int(config["input"]["window_size"])

    stock_history = datas.iloc[0,:,window_size+1:].T
    stock_history['USD'] = np.ones(stock_history.shape[0])

    # stock history
    # need for the date time index to be present
    lastclose = pd.DataFrame(backtest.test_set['lastclose'])
    lastclose.columns = tickers
    lastclose['USD'] = np.ones(lastclose.shape[0])

    # prices per step
    print("test pv :", backtest.test_pc_vector)
    tickers.insert(0,'USD')
    label_history = ['Omega '+ x for x in tickers]
    test_history = pd.DataFrame(backtest.test_omega_vector, columns=label_history)
    test_history = test_history.set_index(stock_history.index.values)
    test_history['pv']=np.cumprod(backtest.test_pc_vector)
    test_history['mu'] = backtest.test_mu_vector
    for ticker in tickers:
        test_history['Share ' + ticker] = test_history['pv'] * test_history['Omega ' + ticker] / stock_history[ticker]
    for ticker in tickers:
        buy_sell = np.diff(test_history['Share ' + ticker])
        test_history['Buy/Sell ' + ticker] = np.insert(buy_sell, 0, test_history['Share '+ticker][0])
    frames = [stock_history, test_history]
    result = pd.concat(frames, axis=1)

    result.to_csv("./train_package/"+algo+"/backtest-"+algo+".csv", mode='w')
    writer = pd.ExcelWriter("./train_package/"+algo+"/backtest-"+algo+".xlsx")
    result.to_excel(writer, 'Backtest')
    writer.save()
    logging.info("finish executing back test")

    print("THIS IS THE END")


def _extract_test(config):
    global_start = parse_time(config["input"]["start_date"])
    global_end = parse_time(config["input"]["end_date"])
    span = global_end - global_start
    start = global_end - datetime.timedelta(int(config["input"]["test_portion"] * span.days))
    end = global_end
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

