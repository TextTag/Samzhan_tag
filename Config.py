# -*- coding: utf-8 -*-
"""
@Time  : 18/7/3 下午7:30 

@author: Samzhanshi
"""
import ConfigParser
import json


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    """General"""
    trainPath = './data_finance/train'
    devPath = './data_finance/dev'
    testPath = './data_finance/test'
    dictPath = './data_finance/dict'
    testRows = './data_finance/tmp_label'
    embed_c_path = None
    embed_w_path = None

    cnn_after_embed = False
    sntEncoder = 'cnn'
    optimizer = 'adam'
    pre_trained = False
    title = True
    content = True
    word_level = False
    vocab_sz_w = 0
    vocab_sz_c = 0
    embed_w = None
    embed_c = None
    embed_size = 200
    batch_size = 64
    max_epochs = 50
    early_stopping = 3
    dropout = 0.9
    lr = 0.001
    select_bar = 1.0
    decay_steps = 500
    decay_rate = 0.9
    class_num = 0
    reg = 1e-5
    num_steps = 40
    fnn_numLayers = 1
    dense_hidden = [200, 2]

    """lstm"""
    hidden_size = 200
    rnn_numLayers = 1

    """cnn"""
    num_filters = 64
    filter_sizes = [3, 4, 5]
    cnn_numLayers = 2


    def saveConfig(self, filePath):
        cfg = ConfigParser.ConfigParser()
        cfg.add_section('General')
        cfg.add_section('lstm')
        cfg.add_section('cnn')

        cfg.set('General', 'trainPath', self.trainPath)
        cfg.set('General', 'devPath', self.devPath)
        cfg.set('General', 'testPath', self.testPath)
        cfg.set('General', 'dictPath', self.dictPath)
        cfg.set('General', 'tmp_label', self.testRows)
        cfg.set('General', 'embed_c_path', self.embed_c_path)
        cfg.set('General', 'embed_w_path', self.embed_w_path)

        cfg.set('General', 'cnn_after_embed', self.cnn_after_embed)
        cfg.set('General', 'sntEncoder', self.sntEncoder)
        cfg.set('General', 'optimizer', self.optimizer)
        cfg.set('General', 'pre_trained', self.pre_trained)
        cfg.set('General', 'title', self.title)
        cfg.set('General', 'content', self.content)
        cfg.set('General', 'word_level', self.word_level)
        cfg.set('General', 'vocab_sz_w', self.vocab_sz_w)
        cfg.set('General', 'vocab_sz_c', self.vocab_sz_c)
        cfg.set('General', 'embed_w', self.embed_w)
        cfg.set('General', 'embed_c', self.embed_c)
        cfg.set('General', 'embed_size', self.embed_size)
        cfg.set('General', 'batch_size', self.batch_size)
        cfg.set('General', 'max_epochs', self.max_epochs)
        cfg.set('General', 'early_stopping', self.early_stopping)
        cfg.set('General', 'dropout', self.dropout)
        cfg.set('General', 'lr', self.lr)
        cfg.set('General', 'decay_steps', self.decay_steps)
        cfg.set('General', 'decay_rate', self.decay_rate)
        cfg.set('General', 'class_num', self.class_num)
        cfg.set('General', 'reg', self.reg)
        cfg.set('General', 'num_steps', self.num_steps)
        cfg.set('General', 'fnn_numLayers', self.fnn_numLayers)
        cfg.set('General', 'dense_hidden', self.dense_hidden)
        cfg.set('General', 'select_bar', self.select_bar)

        cfg.set('lstm', 'hidden_size', self.hidden_size)
        cfg.set('lstm', 'rnn_numLayers', self.rnn_numLayers)

        cfg.set('cnn', 'num_filters', self.num_filters)
        cfg.set('cnn', 'filter_sizes', self.filter_sizes)
        cfg.set('cnn', 'cnn_numLayers', self.cnn_numLayers)

        with open(filePath, 'w') as fd:
            cfg.write(fd)

    def loadConfig(self, filePath):
        cfg = ConfigParser.ConfigParser()
        cfg.read(filePath)

        self.trainPath = cfg.get('General', 'trainPath')
        self.devPath = cfg.get('General', 'devPath')
        self.testPath = cfg.get('General', 'testPath')
        self.dictPath = cfg.get('General', 'dictPath')
        self.testRows = cfg.get('General', 'testRows')
        self.embed_c_path = cfg.get('General', 'embed_c_path')
        self.embed_w_path = cfg.get('General', 'embed_w_path')

        self.cnn_after_embed = cfg.getboolean('General', 'cnn_after_embed')
        self.sntEncoder = cfg.get('General', 'sntEncoder')
        self.optimizer = cfg.get('General', 'optimizer')
        self.pre_trained = cfg.getboolean('General', 'pre_trained')
        self.title = cfg.getboolean('General', 'title')
        self.content = cfg.getboolean('General', 'content')
        self.word_level = cfg.getboolean('General', 'word_level')
        self.vocab_sz_w = cfg.getint('General', 'vocab_sz_w')
        self.vocab_sz_c = cfg.getint('General', 'vocab_sz_c')
        self.embed_w = cfg.get('General', 'embed_w')
        self.embed_c = cfg.get('General', 'embed_c')
        self.embed_size = cfg.getint('General', 'embed_size')
        self.batch_size = cfg.getint('General', 'batch_size')
        self.max_epochs = cfg.getint('General', 'max_epochs')
        self.early_stopping = cfg.getint('General', 'early_stopping')
        self.dropout = cfg.getfloat('General', 'dropout')
        self.lr = cfg.getfloat('General', 'lr')
        self.decay_steps = cfg.getint('General', 'decay_steps')
        self.decay_rate = cfg.getfloat('General', 'decay_rate')
        self.select_bar = cfg.getfloat('General', 'select_bar')
        self.class_num = cfg.getint('General', 'class_num')
        self.reg = cfg.getfloat('General', 'reg')
        self.num_steps = cfg.getint('General', 'num_steps')
        self.fnn_numLayers = cfg.getint('General', 'fnn_numLayers')
        self.dense_hidden = json.loads(cfg.get('General', 'dense_hidden'))

        self.hidden_size = cfg.getint('lstm', 'hidden_size')
        self.rnn_numLayers = cfg.getint('lstm', 'rnn_numLayers')

        self.num_filters = cfg.getint('cnn', 'num_filters')
        self.filter_sizes = json.loads(cfg.get('cnn', 'filter_sizes'))
        self.cnn_numLayers = cfg.getint('cnn', 'cnn_numLayers')