from __future__ import absolute_import, division, print_function
import sys
import time
from builtins import str as text
from datetime import datetime
import json
import os
rootpath = os.path.dirname(os.path.abspath(__file__)).\
    replace("\\thewave\\tools", "").replace("/thewave/tools","")

try:
    unicode        # Python 2
except NameError:
    unicode = str  # Python 3


def preprocess_config(config):
    fill_default(config)
    if sys.version_info[0] == 2:
        return byteify(config)
    else:
        return config


def fill_default(config):
    set_missing(config, "random_seed", 0)
    set_missing(config, "agent_type", "NNAgent")
    fill_layers_default(config["layers"])
    fill_input_default(config["input"])
    fill_train_config(config["training"])

def clean_for_backtest(config):
    config["trading"]["rolling_training_steps"] = 0
    return config

def fill_train_config(train_config):
    set_missing(train_config, "fast_train", True)
    set_missing(train_config, "decay_rate", 1.0)
    try:
        set_missing(train_config, "decay_steps", train_config["steps"]/2)
    finally:
        set_missing(train_config, "decay_steps", 50000)


def fill_input_default(input_config):
    set_missing(input_config, "save_memory_mode", False)
    set_missing(input_config, "portion_reversed", False)
    set_missing(input_config, "market", "poloniex")
    set_missing(input_config, "norm_method", "absolute")
    set_missing(input_config, "is_permed", False)
    set_missing(input_config, "fake_ratio", 1)


def fill_layers_default(layers):
    for layer in layers:
        if layer["type"] == "ConvLayer":
            set_missing(layer, "padding", "valid")
            set_missing(layer, "strides", [1, 1])
            set_missing(layer, "activation_function", "relu")
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "EIIE_Dense":
            set_missing(layer, "activation_function", "relu")
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "DenseLayer":
            set_missing(layer, "activation_function", "relu")
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "EIIE_LSTM" or layer["type"] == "EIIE_RNN":
            set_missing(layer, "dropouts", None)
        elif layer["type"] == "EIIE_Output" or\
                layer["type"] == "Output_WithW" or\
                layer["type"] == "EIIE_Output_WithW":
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "DropOut":
            pass
        elif layer["type"] == "MaxPooling":
            pass
        elif layer["type"] == "BatchNormalization":
            pass
        else:
            raise ValueError("layer name {} not supported".format(layer["type"]))


def set_missing(config, name, value):
    if name not in config:
        config[name] = value


def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return str(input)
    else:
        return input

def parse_list(input):
    """
    return a list based on a string in the json
    of the type "['AAPL','MSFT','ABC','ETN']"
    :param input:
    :return:
    """
    if type(input) == list:
        return input
    else:
        input_quote = input.replace("'","\"")
        list_encoded = json.loads(input_quote)
        ## CAUTION, because of the difference in Python 2 and 3 of str and unicode, we need to change
        ## from builtins import str as text
        return [text(x) for x in list_encoded]
    #return list_encoded
    #return [x.encode('utf-8') for x in list_encoded]


def parse_time(time_string):
    print('time_string ',time_string," : ", datetime.strptime(time_string, "%Y/%m/%d"))
    return datetime.strptime(time_string, "%Y/%m/%d")
    #return time.mktime(datetime.strptime(time_string, "%Y/%m/%d").timetuple())


def load_config(index=None):
    """
    @:param index: if None, load the default in thewave;
     if a integer, load the config under train_package
    """
    print("Process requested :", index)
    if index:
        with open(rootpath+"/train_package/" + str(index) + "/net_config.json") as file:
            config = json.load(file)
    else:
        with open(rootpath+"/thewave/" + "net_config.json") as file:
            config = json.load(file)
    return preprocess_config(config)


def check_input_same(config1, config2):
    input1 = config1["input"]
    input2 = config2["input"]
    if input1["start_date"] != input2["start_date"]:
        return False
    elif input1["end_date"] != input2["end_date"]:
        return False
    elif input1["validation_portion"] != input2["validation_portion"]:
        return False
    elif input1["window_size"] != input2["window_size"]:
        return False
    else:
        return True

