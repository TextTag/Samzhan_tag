# -*- coding: utf-8 -*-
"""
@Time  : 18/7/3 下午7:39

@author: Samzhanshi
This file provides normal training procedure in the father class for inheriting
"""
import os,sys
import numpy as np
import logging
import pandas as pd
import tensorflow as tf
import pickle as p

from util import evaluate, take_out
from data_helpers import BucketedDataIterator as dataSet

class Train_father():

    def __init__(self, Model, Config, args):
        """

        Args:
            model: a Model class, **not an object**
            args: args from bash shell
        """

        config = Config()
        self.args = args
        self.weight_path = args.weight_path
        if args.load_config == False:
            config.saveConfig(self.weight_path + '/config')
            print ('default configuration generated, please specify --load-config and run again.')
            sys.exit()
        else:
            if os.path.exists(self.weight_path + '/config'):
                config.loadConfig(self.weight_path + '/config')
            else:
                config.saveConfig(self.weight_path + '/config')  # if not exists config file then use default

        self.data_train = dataSet(pd.read_csv(config.trainPath))
        self.data_dev = dataSet(pd.read_csv(config.devPath))
        self.data_test = dataSet(pd.read_csv(config.testPath))

        with open(config.dictPath,'rb') as f:
            text = p.load(f)
        config.vocab_sz_w = len(text)
        config.class_num = 2

        self.config = config
        self.model = Model(self.config)

    def run_epoch(self, sess, verbose=10):
        """Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
            data_x: input data, have shape of (data_num, num_steps), change it to ndarray before this function is called
            data_y: label, have shape of (data_num, class_num)
            len_list: length list correspond to data_x, have shape of (data_num)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        data = self.data_train
        data_len = data.total
        total_steps = data_len // self.config.batch_size
        total_loss = []
        for step in range(total_steps):
            data_batch = data.next_batch(self.config.batch_size)
            feed_dict = self.model.create_feed_dict(data_batch, train=True)
            _, loss, lr = sess.run([self.model.train_op, self.model.loss,
                                    self.model.learning_rate],
                                   feed_dict=feed_dict)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}, lr = {}'.format(
                    step, total_steps, np.mean(total_loss[-verbose:]), lr))
                sys.stdout.flush()
                # write_status(self.weight_path)

        return np.mean(total_loss)

    def fit(self, sess):
        data = self.data_dev

        data_len = data.total
        total_steps = data_len // self.config.batch_size
        total_loss = []
        for step in range(total_steps):
            data_batch = data.next_batch(self.config.batch_size)
            feed_dict = self.model.create_feed_dict(data_batch, train=False)
            loss = sess.run(self.model.loss, feed_dict=feed_dict)
            total_loss.append(loss)
        return np.mean(total_loss)

    def predict(self, sess, mode='dev'):
        """Make predictions from the provided model.
        Args:
            sess: tf.Session()
            mode: on test set or dev set
        Returns:
          ret_pred_prob: Probability of the prediction with respect to each class
        """

        if mode == 'dev':
            data = self.data_dev
        else:
            data = self.data_test

        data_len = data.total
        total_steps = data_len // self.config.batch_size
        ret_title_pred_prob = []
        ret_content_pred_prob = []
        # ret_title_lens = []
        # ret_content_lens = []
        for step in range(total_steps+1):
            if step == total_steps:
                batch_size = data_len - self.config.batch_size * total_steps
            else:
                batch_size = self.config.batch_size
            if mode == 'dev':
                data_batch = data.next_all_batch(batch_size)
            else:
                data_batch = data.next_all_batch(batch_size, True)
            # ret_title_len, ret_con_len = data_batch[3], data_batch[4]
            feed_dict = self.model.create_feed_dict(data_batch, train=False)
            predict_prob_tit, predict_prob_con, \
                = sess.run([self.model.predict_tit_prob, self.model.predict_con_prob,
                            ], feed_dict=feed_dict)
            ret_title_pred_prob.extend(np.split(predict_prob_tit, batch_size))
            ret_content_pred_prob.extend(np.split(predict_prob_con, batch_size))
        # ret_title_pred_prob = np.concatenate(ret_title_pred_prob, axis=0)
        # ret_content_pred_prob = np.concatenate(ret_content_pred_prob, axis=0)
        return ret_title_pred_prob, ret_content_pred_prob

    def test_case(self, sess, onset='VALIDATION'):
        if onset == 'VALIDATION':
            print ('#' * 20, 'ON ' + onset + ' SET START ', '#' * 20)
            loss = self.fit(sess)
            pred_prob = self.predict(sess, mode='dev')
            accuracy = evaluate(pred_prob, self.data_dev.get_src(), self.data_dev.get_answer(), self.config.select_bar)
            print ('Overall ' + onset + ' accuracy is: {}'.format(accuracy))
            logging.info('Overall ' + onset + ' accuracy is: {}'.format(accuracy))
            print ('Overall ' + onset + ' loss is: {}'.format(loss))
            logging.info('Overall ' + onset + ' loss is: {}'.format(loss))
            print ('#' * 20, 'ON ' + onset + ' SET END ', '#' * 20)
            return accuracy, loss
        else:
            print ('#' * 20, 'ON ' + onset + ' SET START ', '#' * 20)
            pred_prob = self.predict(sess, mode='test')
            take_out(pred_prob, self.data_test.get_src())
            print ('#' * 20, 'ON ' + onset + ' SET END ', '#' * 20)


    def train_run(self):
        logging.info('Training start')
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            best_accuracy = 0
            best_val_epoch = 0
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.config.max_epochs):
                print ("=" * 20 + "Epoch ", epoch, "=" * 20)
                loss = self.run_epoch(sess)
                print
                print ("Mean loss in this epoch is: ", loss)
                logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss))
                print ('=' * 50)

                val_accuracy, loss = self.test_case(sess, onset='VALIDATION')

                if best_accuracy < val_accuracy[2]:
                    best_accuracy = val_accuracy[2]
                    best_val_epoch = epoch
                    if not os.path.exists(self.weight_path):
                        os.makedirs(self.weight_path)

                    saver.save(sess, self.weight_path + '/classifier.weights')
                if epoch - best_val_epoch > self.config.early_stopping:
                    logging.info("Normal Early stop")
                    break

        logging.info("Training complete")

    def test_run(self, mode):

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.weight_path + '/classifier.weights')
            self.test_case(sess, onset=mode)

    def main_run(self):
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
        logFile = self.weight_path + '/run.log'
        if self.args.train_test == "train":
            try:
                os.remove(logFile)
            except OSError:
                pass
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
            self.train_run()
            self.test_run('VALIDATION')
        else:
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
            self.test_run('TEST')
