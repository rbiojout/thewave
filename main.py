from __future__ import absolute_import
import json
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from thewave.tools.configprocess import preprocess_config, clean_for_backtest
from thewave.tools.configprocess import load_config
from thewave.tools.shortcut import execute_backtest
from thewave.resultprocess import plot


from thewave.marketdata.globaldatamatrix import HistoryManager
from thewave.marketdata.datamatrices import DataMatrices
from datetime import date

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="start mode, train, generate, download_data"
                             " backtest, plot, table, resultfile",
                        metavar="MODE", default="train")
    parser.add_argument("--processes", dest="processes",
                        help="number of processes you want to start to train the network",
                        default="1")
    parser.add_argument("--repeat", dest="repeat",
                        help="repeat times of generating training subfolder",
                        default="1")
    parser.add_argument("--algo",
                        help="single algo name or indexes of training_package ",
                        dest="algo")
    parser.add_argument("--algos",
                        help="list of algo names or indexes of training_package, seperated by \",\"",
                        dest="algos")
    parser.add_argument("--labels", dest="labels",
                        help="names that will shown in the figure caption or table header")
    parser.add_argument("--format", dest="format", default="raw",
                        help="format of the table printed")
    parser.add_argument("--device", dest="device", default="cpu",
                        help="device to be used to train")
    parser.add_argument("--folder", dest="folder", type=int,
                        help="folder(int) to load the config, neglect this option if loading from ./thewave/net_config")
    return parser




def main():
    parser = build_parser()
    options = parser.parse_args()

    if options.mode == "train":
        import thewave.autotrain.training
        if not options.algo:
            print('START TRAIN')
            thewave.autotrain.training.train_all(int(options.processes), options.device)
            print('END TRAIN')
        else:
            for folder in options.train_folder:
                raise NotImplementedError()
    elif options.mode == "generate":
        import thewave.autotrain.generate as generate
        logging.basicConfig(level=logging.INFO)
        generate.add_packages(load_config(), int(options.repeat))
    elif options.mode == "download_data":
        from thewave.marketdata.datamatrices import DataMatrices

        with open("./thewave/net_config.json") as file:
            config = json.load(file)
        config = preprocess_config(config)
        dm = DataMatrices.create_from_config(config)
        print("DM global_matrix = ", dm.global_matrix)
        print("DM global_weights = ", dm.global_weights)
        print("dm.ticker_list ", dm.ticker_list)
        print('num_train_samples ', dm.num_train_samples)
        print('num_validation_samples ', dm.num_validation_samples)
        print('validation_indices ', dm.validation_indices)

        print('get_validation_set ', dm.get_validation_set()['X'].shape)
        print('get_training_set ', dm.get_training_set()['X'].shape)
        print('END')
    elif options.mode == "backtest":
        config = _config_by_algo(options.algo)
        _set_logging_by_algo(logging.DEBUG, logging.DEBUG, options.algo, "backtestlog")
        execute_backtest(options.algo, config)
    # python main.py --mode=plot --algo=5 --algos=5,olmar,ons,ubah --labels=nntrader,olmar,ons,ubah
    elif options.mode == "plot":
        logging.basicConfig(level=logging.INFO)
        algo = options.algo
        algos = options.algos.split(",")
        if options.labels:
            labels = options.labels.replace("_"," ")
            labels = labels.split(",")
        else:
            labels = algos
        config = load_config(algo)
        config = clean_for_backtest(config)
        plot.plot_backtest(config, algos, labels)
    # python main.py --mode=table  --algo=5 --algos=5,olmar,ons,ubah --labels=nntrader,olmar,ons,ubah
    elif options.mode == "table":
        logging.basicConfig(level=logging.DEBUG)
        algos = options.algos.split(",")
        algo = options.algo
        if options.labels:
            labels = options.labels.replace("_"," ")
            labels = labels.split(",")
        else:
            labels = algos
        config = load_config(algo)
        config = clean_for_backtest(config)
        plot.table_backtest(config, algos, labels, format=options.format)
    elif options.mode == "resultfile":
        logging.basicConfig(level=logging.DEBUG)
        algo = options.algo
        config = load_config(algo)
        config = clean_for_backtest(config)
        plot.file_backtest(config, algo)

def _set_logging_by_algo(console_level, file_level, algo, name):
    if algo.isdigit():
            logging.basicConfig(filename="./train_package/"+algo+"/"+name,
                                level=file_level)
            console = logging.StreamHandler()
            console.setLevel(console_level)
            logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(level=console_level)


def _config_by_algo(algo):
    """
    :param algo: a string represent index or algo name
    :return : a config dictionary
    """
    if not algo:
        raise ValueError("please input a specific algo")
    elif algo.isdigit():
        config = load_config(algo)
    else:
        config = load_config()
    return config

if __name__ == "__main__":
    main()
