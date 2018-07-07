# -*- coding: utf-8 -*-
"""
@Time  : 18/7/3 下午7:39

@author: Samzhanshi
This class is to provide specific model for our task
1. We use a model based on sequence labeling.
2. It mainly includes feed and add model
"""
import tensorflow as tf

from Model_father import Model_father
import TfUtils

class Model(Model_father, object):
    """
    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs.
    """
    def __init__(self, config):
        super(Model, self).__init__(config)
        loss, (logits_tit, logits_con) = self.add_model()
        self.train_op = self.add_train_op(loss)
        self.loss = loss
        self.predict_tit_prob = tf.nn.softmax(logits_tit)
        self.predict_con_prob = tf.nn.softmax(logits_con)



    def create_feed_dict(self, input_batch, train=True):
        if len(input_batch) == 7:
            holder_list = [self.ph_q_category_w, self.ph_q_title_w, self.ph_q_content_w,
                           self.ph_seqLen_q_title_w, self.ph_seqLen_q_content_w, self.ph_title_label,
                           self.ph_content_label, self.ph_dropout, self.ph_train]
        else:
            holder_list = [self.ph_q_category_w, self.ph_q_title_w, self.ph_q_content_w,
                           self.ph_seqLen_q_title_w, self.ph_seqLen_q_content_w,
                           self.ph_dropout, self.ph_train]

        feed_list = input_batch + (self.config.dropout, train)
        feed_dict = dict(zip(holder_list, feed_list))
        return feed_dict

    def add_model(self):

        def get_representation():
            '''
            Get the representation for final classification, note

            Returns:
                output_for_title: shape(b_sz, rep_sz)(b_sz, seq_title, rep_sz)
                output_for_content: shape(b_sz, seq_content, rep_sz)
            '''
            if self.config.sntEncoder == 'lstm':
                seqEncoder = lambda seqInput, seqLen: self.snt_encoder_lstm(seqInput, seqLen)
            elif self.config.sntEncoder == 'lstm_avg':
                seqEncoder = lambda seqInput, seqLen: self.snt_encoder_lstm_avg(seqInput, seqLen)
            elif self.config.sntEncoder == 'bilstm':
                seqEncoder = lambda seqInput, seqLen: self.snt_encoder_bilstm(seqInput, seqLen)
            elif self.config.sntEncoder == 'bilstm_avg':
                seqEncoder = lambda seqInput, seqLen: self.snt_encoder_bilstm_avg(seqInput, seqLen)
            elif self.config.sntEncoder == 'cbow':
                seqEncoder = lambda seqInput, seqLen: self.snt_encoder_cbow(seqInput, seqLen)
            elif self.config.sntEncoder == 'cnn':
                seqEncoder = lambda seqInput, seqLen: self.snt_encoder_cnn(seqInput, seqLen)
            else:
                raise ValueError('no such encoder: %s' % self.config.sntEncoder)
            seqInputs = []
            seqLens = []
            # seqInputs[0] is title, [1] is content
            if self.config.title:
                seqInputs.append(self.q_title_w_input)
                seqLens.append(self.ph_seqLen_q_title_w)
            if self.config.content:
                seqInputs.append(self.q_content_w_input)
                seqLens.append(self.ph_seqLen_q_content_w)
            if len(seqLens) == 0 or len(seqInputs) == 0:
                raise ValueError('must set config.title or config.content to True')

            inputs = zip(seqInputs, seqLens)
            with tf.variable_scope('snt_encoder') as scp:
                enc_snt = []
                enc_pos = []
                for i, item in enumerate(inputs): # shape(b_sz, title_fetch_sz(or +)desp_fetch_sz)
                    if i == 1:
                        scp.reuse_variables()
                    snt, pos = seqEncoder(*item)
                    enc_snt.append(snt)
                    enc_pos.append(pos)
                output = tf.concat(enc_snt, axis=1)
            # enc_pos[0] means title enc_pos[1] means content

            output = tf.concat([output, self.q_category_w_input], axis=1)
            output_for_title = tf.tile(tf.expand_dims(output, 1), [1, tf.reduce_max(inputs[0][1]), 1])
            output_for_content = tf.tile(tf.expand_dims(output, 1), [1, tf.reduce_max(inputs[1][1]), 1])
            output_for_title = tf.concat([output_for_title, enc_pos[0]], axis=2)
            output_for_content = tf.concat([output_for_content, enc_pos[1]], axis=2)
            # Experiment Part
            output_for_title = tf.concat([output_for_title, seqInputs[0]], axis=2)
            output_for_content = tf.concat([output_for_content, seqInputs[1]], axis=2)
            return output_for_title, output_for_content


        def Dense(output_for_title, output_for_content):
            '''
            Get the logits for final classification, note

            Returns:
                output_for_title: shape(b_sz, rep_sz)(b_sz, seq_title, class_num)
                output_for_content: shape(b_sz, seq_content, class_num)
            '''
            batch_size = tf.shape(output_for_title)[0]
            # batch_dim = self.config.embed_size + self.config.num_filters * len(self.config.filter_sizes) * 3
            batch_dim = 2 * self.config.embed_size + self.config.num_filters * len(self.config.filter_sizes) * 3
            print (batch_dim)

            loop_input_title = tf.reshape(output_for_title, [-1, batch_dim])
            loop_input_content = tf.reshape(output_for_content, [-1, batch_dim])
            if self.config.dense_hidden[-1] != self.config.class_num:
                raise ValueError('last hidden layer should be %d, but get %d'%
                                 (self.config.class_num,
                                 self.config.dense_hidden[-1]))
            for i, hid_num in enumerate(self.config.dense_hidden):
                loop_input_title = TfUtils.linear(loop_input_title, output_size=hid_num,
                                            bias=True, scope='dense-tit-layer-%d'%i)
                if i < len(self.config.dense_hidden)-1:
                    loop_input_title = tf.nn.relu(loop_input_title)

                loop_input_content = TfUtils.linear(loop_input_content, output_size=hid_num,
                                                  bias=True, scope='dense-con-layer-%d' % i)
                if i < len(self.config.dense_hidden) - 1:
                    loop_input_content = tf.nn.relu(loop_input_content)

            logits = (tf.reshape(loop_input_title, [batch_size, -1, self.config.class_num]), tf.reshape(loop_input_content, [batch_size, -1, self.config.class_num]))
            return logits

        def add_loss_op(logits, title_label, content_label, tit_len, content_len):
            '''
            Returns:
                loss
            '''
            title_logits, content_logits = logits

            loss1 = TfUtils.seq_loss(title_logits, title_label, tit_len)
            loss2 = TfUtils.seq_loss(content_logits, content_label, content_len)

            loss = tf.reduce_mean(loss1 + loss2)
            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                 if v not in [self.embed_w]])
            return loss + self.config.reg * reg_loss

        title_rep, content_rep = get_representation()
        logits = Dense(title_rep, content_rep)

        loss = add_loss_op(logits, self.ph_title_label, self.ph_content_label, self.ph_seqLen_q_title_w, self.ph_seqLen_q_content_w)
        return loss, logits