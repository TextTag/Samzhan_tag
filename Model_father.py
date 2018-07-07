# -*- coding: utf-8 -*-
"""
@Time  : 18/7/3 下午7:39

@author: Samzhanshi
This class is to provide father model class for inheriting
1. It gives general interface, like placeholder embedding, training_op
2. It includes all general sentence modeling tech
"""

import tensorflow as tf

import TfUtils
# import core_rnn_cell_impl as rnn_cell
import tensorflow.contrib.rnn as rnn_cell

class Model_father():
    """
    Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """

    def __init__(self, config):
        self.config = config
        self.embed_w = self.__add_embedding(self.config.vocab_sz_w, self.config.embed_w, name='embed_w')

        self.add_placeholders()
        with tf.variable_scope('fetch_input') as scp:
            self.q_title_w_input = self.fetch_input(self.embed_w, self.ph_q_title_w, scope='fetch_w')
            scp.reuse_variables()
            self.q_content_w_input = self.fetch_input(self.embed_w, self.ph_q_content_w, scope='fetch_w')
            self.q_category_w_input = tf.nn.embedding_lookup(self.embed_w, self.ph_q_category_w)

    def add_placeholders(self):
        self.ph_q_title_w = tf.placeholder(tf.int32, (None, None), name='ph_q_title_w')
        self.ph_q_content_w = tf.placeholder(tf.int32, (None, None), name='ph_q_content_w')
        self.ph_q_category_w = tf.placeholder(tf.int32, (None,), name='ph_t_cate_w')

        self.ph_seqLen_q_title_w = tf.placeholder(tf.int32, (None,), name='ph_seqLen_q_title_w')
        self.ph_seqLen_q_content_w = tf.placeholder(tf.int32, (None,), name='ph_seqLen_q_content_w')

        self.ph_title_label = tf.placeholder(tf.int32, (None, None), name='ph_title_label')
        self.ph_content_label = tf.placeholder(tf.int32, (None, None), name='ph_label_label')
        self.ph_dropout = tf.placeholder(tf.float32, name='ph_dropout')
        self.ph_train = tf.placeholder(tf.bool, name='ph_train')

    def __add_embedding(self, vocab_sz, embed_matrix, name):
        if self.config.pre_trained:
            embedding = tf.Variable(embed_matrix, name=name)
        else:
            embedding = tf.get_variable(name, shape=[vocab_sz, self.config.embed_size],initializer=tf.random_normal_initializer(
            mean=0.0, stddev=0.1),trainable=True)
        return embedding

    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.lr, global_step,
                                                   self.config.decay_steps, self.config.decay_rate, staircase=True)

        if self.config.optimizer == 'adgrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError('No such optimizer: %s' % self.config.optimizer)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def fetch_input(self, embedding, seqIds, scope):
        '''

        Args:
            embedding: embedding matrix to lookup from
            seqIds: sequence ids

        Returns:
            output: shape(b_sz, maxSeqLen, fetch_h_sz)
        '''
        inputs = tf.nn.embedding_lookup(embedding, seqIds)  # shape(b_sz, tstp, emb_sz)
        if self.config.cnn_after_embed:
            with tf.variable_scope('cnn_after_embed_%s' %scope):
                filter_shape = [3, self.config.embed_size, self.config.embed_size]
                W = tf.get_variable(name='W', shape=filter_shape)
                b = tf.get_variable(name='b', shape=[self.config.embed_size])
                conv = tf.nn.conv1d(  # size (b_sz, tstp, out_channel)
                    inputs,
                    W,
                    stride=1,
                    padding="SAME",
                    name="conv")

            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            inputs = h          # shape(b_sz, tstp, emb_sz)
        inputs = TfUtils.Dropout(inputs, self.config.dropout, train=self.ph_train)

        return inputs

    def snt_encoder_lstm(self, seqInput, seqLen):
        '''
        Take the last hidden state of lstm as sentence representation

        Args:
            seqInput: encoder input, shape(b_sz, maxSeqLen, dim_x)
            seqLen:   length for each sequence in the batch

        Returns:
            output: shape(b_sz, dim_h)
        '''

        lstm_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)

        output, states = tf.nn.dynamic_rnn(cell=lstm_cell,inputs=seqInput,
                                           sequence_length=seqLen, dtype=tf.float32,
                                           swap_memory=True, scope='snt_enc')

        snt_enc = states[1]
        return snt_enc

    def snt_encoder_lstm_avg(self, seqInput, seqLen):
        '''
        Take the average of output as sentence representation

        Args:
            seqInput: encoder input, shape(b_sz, maxSeqLen, dim_x)
            seqLen:   length for each sequence in the batch

        Returns:
            output: shape(b_sz, dim_h)
        '''
        lstm_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)
        output, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=seqInput,
                                           sequence_length=seqLen, dtype=tf.float32,
                                           swap_memory=True, scope='snt_enc')
        snt_enc = TfUtils.reduce_avg(output, lengths=seqLen, dim=1)
        return snt_enc

    def snt_encoder_bilstm(self, seqInput, seqLen):
        '''
        Take the last hidden state of bilstm as sentence representation

        Args:
            seqInput: encoder input, shape(b_sz, maxSeqLen, dim_x)
            seqLen:   length for each sequence in the batch

        Returns:
            output: shape(b_sz, dim_h)
        '''
        lstm_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)

        output, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell, cell_bw=lstm_cell,
                                                         inputs=seqInput, sequence_length=seqLen,
                                                         dtype=tf.float32, swap_memory=True,
                                                         scope='snt_enc')

        states = [states[0][1], states[1][1]]
        snt_enc = tf.concat(states, axis=1)
        return snt_enc

    def snt_encoder_bilstm_avg(self, seqInput, seqLen):
        '''
        Take the average of output state of bilstm as sentence representation

        Args:
            seqInput: encoder input, shape(b_sz, maxSeqLen, dim_x)
            seqLen:   length for each sequence in the batch

        Returns:
            output: shape(b_sz, dim_h)
        '''

        lstm_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)
        output, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell, cell_bw=lstm_cell,
                                                         inputs=seqInput, sequence_length=seqLen,
                                                         dtype=tf.float32, swap_memory=True,
                                                         scope='snt_enc')
        output = tf.concat(output, axis=2)
        snt_enc = TfUtils.reduce_avg(output, lengths=seqLen, dim=1)
        return snt_enc

    def snt_encoder_cbow(self, seqInput, seqLen):
        '''
        Take the average word representation as sentence representation

        Args:
            seqInput: encoder input, shape(b_sz, maxSeqLen, dim_x)
            seqLen:   length for each sequence in the batch

        Returns:
            output: shape(b_sz, dim_h)
        '''

        aggregate_state = TfUtils.reduce_avg(seqInput, seqLen, dim=1)  # b_sz, emb_sz

        return aggregate_state

    # Modify this method temporarily
    def snt_encoder_cnn(self, seqInput, seqLen):
        '''
        CNN encoder

        Args:
            seqInput: encoder input, shape(b_sz, maxSeqLen, dim_x)
            seqLen:   length for each sequence in the batch

        Returns:
            output: shape(b_sz, dim_h)
        '''
        input_shape = tf.shape(seqInput)
        b_sz = input_shape[0]
        tstp = input_shape[1]

        in_channel = self.config.embed_size
        filter_sizes = self.config.filter_sizes
        out_channel = self.config.num_filters
        input = seqInput
        for layer in range(self.config.cnn_numLayers):
            with tf.variable_scope("conv-layer-" + str(layer)):
                conv_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.variable_scope("conv-maxpool-%d" % filter_size):
                        # Convolution Layer
                        filter_shape = [filter_size, in_channel, out_channel]
                        W = tf.get_variable(name='W', shape=filter_shape)
                        b = tf.get_variable(name='b', shape=[out_channel])
                        conv = tf.nn.conv1d(  # size (b_sz, tstp, out_channel)
                            input,
                            W,
                            stride=1,
                            padding="SAME",
                            name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        conv_outputs.append(h)
                input = tf.concat(axis=2, values=conv_outputs)  # b_sz, tstp, out_channel*len(filter_sizes)
                in_channel = out_channel * len(filter_sizes)

        mask = TfUtils.mkMask(seqLen, tstp)  # b_sz, tstp

        pooled = tf.reduce_mean(input * tf.expand_dims(tf.cast(mask, dtype=tf.float32), 2), [1])
        # size (b_sz, out_channel*len(filter_sizes))
        snt_enc = tf.reshape(pooled, shape=[b_sz, out_channel * len(filter_sizes)])
        return snt_enc, input
