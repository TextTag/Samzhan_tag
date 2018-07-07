# -*- coding: utf-8 -*-
"""
@Time  : 18/7/3 下午7:39

@author: Samzhanshi
Training file
"""
import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from Train_father import Train_father
from SeqLabel_model import Model
from Config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser(description="training options")

parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)
parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
args = parser.parse_args()


class Train(Train_father, object):
    def __init__(self):
        super(Train, self).__init__(Model, Config, args)

if __name__ == '__main__':
    trainer = Train()
    trainer.main_run()
