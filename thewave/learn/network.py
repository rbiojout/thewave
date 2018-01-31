#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import tflearn


class NeuralNetWork:
    def __init__(self, feature_number, rows, columns, layers, device):
        tf_config = tf.ConfigProto()
        self.session = tf.Session(config=tf_config)
        if device == "cpu":
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        else:
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.input_num = tf.placeholder(tf.int32, shape=[], name='input_num')
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, feature_number, rows, columns], name='input_tensor')
        self.previous_w = tf.placeholder(tf.float32, shape=[None, rows], name='previous_w')
        self._rows = rows
        self._columns = columns
        self.output = self._build_network(layers)

    def _build_network(self, layers):
        pass


class CNN(NeuralNetWork):
    # input_shape (features, rows=assets, columns=batch)
    def __init__(self, feature_number, rows, columns, layers, device):
        NeuralNetWork.__init__(self, feature_number, rows, columns, layers, device)

    # generate the operation, the forward computation
    def _build_network(self, layers):
        network = tf.transpose(self.input_tensor, [0, 2, 3, 1])
        # [batch, assets, window, features]
        network = network / network[:, :, -1, 0, None, None]
        for layer_number, layer in enumerate(layers):
            if layer["type"] == "DenseLayer":
                network = tflearn.layers.core.fully_connected(network,
                                                              int(layer["neuron_number"]),
                                                              layer["activation_function"],
                                                              regularizer=layer["regularizer"],
                                                              weight_decay=layer["weight_decay"] )
            elif layer["type"] == "DropOut":
                network = tflearn.layers.core.dropout(network, layer["keep_probability"])
            elif layer["type"] == "EIIE_Dense":
                width = network.get_shape()[2]
                # default for conv2d : weights_init='truncated_normal', regularizer=None, weight_decay=0.001
                network = tflearn.layers.conv_2d(network, nb_filter=int(layer["filter_number"]),
                                                 filter_size=[1, width],
                                                 strides=[1, 1],
                                                 padding="valid",
                                                 activation=layer["activation_function"],
                                                 bias=True,
                                                 weights_init='uniform_scaling',
                                                 bias_init='zeros',
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"],
                                                 trainable=True, restore=True, reuse=False, scope=None,
                                                 name="EIIE_Dense")
            elif layer["type"] == "ConvLayer":
                # default for conv2d : weights_init='truncated_normal', regularizer=None, weight_decay=0.001
                network = tflearn.layers.conv_2d(network, nb_filter=int(layer["filter_number"]),
                                                 filter_size=allint(layer["filter_shape"]),
                                                 strides=allint(layer["strides"]),
                                                 padding=layer["padding"],
                                                 activation=layer["activation_function"],
                                                 bias=True,
                                                 weights_init='uniform_scaling',
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"],
                                                 trainable=True, restore=True, reuse=False, scope=None,
                                                 name="ConvLayer2D")
            elif layer["type"] == "MaxPooling":
                network = tflearn.layers.conv.max_pool_2d(network, layer["strides"])
            elif layer["type"] == "AveragePooling":
                network = tflearn.layers.conv.avg_pool_2d(network, layer["strides"])
            elif layer["type"] == "BatchNormalization":
                network = tflearn.layers.normalization.batch_normalization(network, restore=True)
            elif layer["type"] == "LocalResponseNormalization":
                network = tflearn.layers.normalization.local_response_normalization(network)
            elif layer["type"] == "EIIE_Output":
                width = network.get_shape()[2]
                # default for conv2d : weights_init='truncated_normal',regularizer=None, weight_decay=0.001
                network = tflearn.layers.conv_2d(network, nb_filter=1,
                                                 filter_size=[1, width],
                                                 strides=1,
                                                 padding="valid",
                                                 activation='linear', bias=True,
                                                 # weights_init='uniform_scaling',
                                                 weights_init='uniform',
                                                 bias_init='zeros',
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"],
                                                 trainable=True, restore=True, reuse=False, scope=None,
                                                 name="EIIE_Output")
                network = network[:, :, 0, 0]
                btc_bias = tf.ones((self.input_num, 1))
                network = tf.concat([btc_bias, network], 1)
                network = tflearn.layers.core.activation(network, activation="softmax")
            elif layer["type"] == "Output_WithW":
                network = tflearn.flatten(network)
                network = tf.concat([network,self.previous_w], axis=1)
                # default for fully_connected : regularizer=None, weight_decay=0.001
                network = tflearn.fully_connected(network, n_units=self._rows+1,
                                                  activation="softmax",
                                                  bias=True,
                                                  weights_init='truncated_normal', bias_init='zeros',
                                                  regularizer=layer["regularizer"],
                                                  weight_decay=layer["weight_decay"],
                                                  trainable=True,
                                                  restore=True, reuse=False, scope=None,
                                                  name="Output_WithW"                                                  )
            elif layer["type"] == "EIIE_Output_WithW":
                width = network.get_shape()[2]
                height = network.get_shape()[1]
                features = network.get_shape()[3]
                network = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                w = tf.reshape(self.previous_w, [-1, int(height), 1, 1])
                network = tf.concat([network, w], axis=3)
                # default for conv2d : regularizer=None, weight_decay=0.001
                network = tflearn.layers.conv_2d(network, nb_filter=1, filter_size=[1, 1],
                                                 strides=1,
                                                 padding="valid",
                                                 activation='linear', bias=True,
                                                 weights_init='uniform_scaling',
                                                 bias_init='zeros',
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"],
                                                 trainable=True, restore=True, reuse=False, scope=None,
                                                 name="EIIE_Output_WithW")
                network = network[:, :, 0, 0]
                #btc_bias = tf.zeros((self.input_num, 1))
                btc_bias = tf.get_variable("btc_bias", [1, 1], dtype=tf.float32,
                                            initializer = tf.zeros_initializer)
                btc_bias = tf.tile(btc_bias, [self.input_num, 1])
                network = tf.concat([btc_bias, network], 1)
                self.voting = network
                network = tflearn.layers.core.activation(network, activation="softmax")

            elif layer["type"] == "EIIE_LSTM" or\
                            layer["type"] == "EIIE_RNN":
                network = tf.transpose(network, [0, 2, 3, 1])
                resultlist = []
                reuse = False
                for i in range(self._rows):
                    if i > 0:
                        reuse = True
                    if layer["type"] == "EIIE_LSTM":
                        result = tflearn.layers.lstm(network[:, :, :, i],
                                                     n_units=int(layer["neuron_number"]),
                                                     activation='tanh', inner_activation='sigmoid',
                                                     dropout=layer["dropouts"],
                                                     bias=True, weights_init=None, forget_bias=1.0,
                                                     return_seq=False, return_state=False, initial_state=None,
                                                     dynamic=False, trainable=True, restore=True,
                                                     scope="lstm"+str(layer_number),
                                                     reuse=reuse,
                                                     name="LSTM")
                    else:
                        result = tflearn.layers.simple_rnn(network[:, :, :, i],
                                                           n_units=int(layer["neuron_number"]),
                                                           activation='sigmoid',
                                                           dropout=layer["dropouts"],
                                                           bias=True, weights_init=None, return_seq=False,
                                                           return_state=False, initial_state=None, dynamic=False,
                                                           trainable=True, restore=True,
                                                           scope="rnn"+str(layer_number),
                                                           reuse=reuse,
                                                           name="SimpleRNN")
                    resultlist.append(result)
                network = tf.stack(resultlist)
                network = tf.transpose(network, [1, 0, 2])
                network = tf.reshape(network, [-1, self._rows, 1, int(layer["neuron_number"])])
            else:
                raise ValueError("the layer {} not supported.".format(layer["type"]))
        return network


def allint(l):
    return [int(i) for i in l]
