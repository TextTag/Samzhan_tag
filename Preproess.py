# -*- coding: utf-8 -*-
"""
@Time  : 18/7/3 上午11:03 

@author: Samzhanshi

This module is for the information extraction of the excel. i.e. Convert a data file
in excel format into csv format with vocabulary built. The extraction processes are
as follows:
1. Create User dict, which includes all labels tagged over the whole documents
2. Segment titles and documents, use the segmented words to build vocab
   Create dict to collect information by domain
3. Split train and test and build the csv file, which includes all necessary information
   for deep learning models

"""

import xlrd
from tqdm import tqdm
import codecs
import jieba.posseg as pseg

import re
import jieba
import csv
from collections import defaultdict
from collections import Counter
import pickle
import csv
import argparse
import pandas as pd
import os

# These regs are for match and replace
rURL = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
rDEL = '#'
rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
rENG = '[A-Za-z_.]+'


class Data_extraction(object):
    def __init__(self, inpath, sheetname, usr_dict, stop_words, domain, test_data, train):
        self.sheetname = sheetname
        self.usr_dict = usr_dict
        self.stop_words = stop_words
        dirs = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(dirs, 'data_'+str(domain))
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        user_dict = os.path.join(dirs, usr_dict)
        self.raw_data = os.path.join(dirs, inpath)
        self.test_data = os.path.join(dirs, test_data)
        self.train_out = os.path.join(self.data_dir, 'train')
        self.dev_out = os.path.join(self.data_dir, 'dev')
        self.test_out = os.path.join(self.data_dir, 'test')

        self.domain_dict = os.path.join(self.data_dir, 'dict')
        if not os.path.exists(user_dict):
            self.create_usr_dict()
        if train:
            tok_set, words_set = self.generator()
            vocab, words= self.generate_vocab(words_set)
            self.write2csv(tok_set, vocab)
        else:
            with open(self.domain_dict,'rb') as f:
                text = pickle.load(f)
            vocabulary = {x: i for i, x in enumerate(text)}
            self.domain_set = self.collect_train_domains()
            token_set = self.generate_test()
            self.write2csv(token_set, vocabulary, False)

    def tokenization(self, title):
        '''
        :param title: String in the cell of excel
        :return: A string after seg and replace, remove stop words
        '''
        result = []
        # stop_words = './stop_words.txt'
        stopwords = codecs.open(self.stop_words, 'r', encoding='utf8').readlines()
        stopwords = set([w.strip() for w in stopwords])
        # stop_flag = set(['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r'])
        title = re.sub(rURL, '', title)
        title = re.sub(rDEL, '', title)
        # title = re.sub(rENG, 'X', title)
        words = pseg.cut(title)
        for word, flag in words:
            # if flag not in stop_flag and word not in stopwords:
            if word not in stopwords:
                result.append(word)
        return result

    def generate_vocab(self, words_set):
        '''
        :param words_set: a list of word
        :return: create a w2idx dict and save the vocab
        We calculate the word frequency, take the top 40000 words as well as
        words in labels as vocab.
        '''
        word_counts = Counter(words_set)
        vocabulary_inv = [x[0] for x in word_counts.most_common(40000)]
        with codecs.open(self.usr_dict, 'r', 'utf-8') as f:
            text = f.readlines()
        print ('Usr dict size', len(text))
        vocabulary_inv = list(sorted(vocabulary_inv))
        vocabulary_inv.extend([word.strip() for word in text])
        vocabulary_inv = list(set(vocabulary_inv))
        print ('vocab_size', len(vocabulary_inv))
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        print ('UNK position', vocabulary['UNK'])
        with open(self.domain_dict, 'wb') as f:
            pickle.dump(vocabulary_inv, f)

        return [vocabulary, vocabulary_inv]

    def to_idx(self, lst, dic):
        '''
        :param lst: one line of csv file in list format
        'class', 'titles', 'tit_len', 'content', 'cnt_len', 'labels', 'indices_title', 'indices_content'
        :param dic: vocabulary
        :return: a csv file with indexes of the words
        '''
        all = []
        for idx, elem in enumerate(lst):
            tmp = []
            if idx == 2 or idx == 4:
                tmp.append(elem)
            elif idx == 0 or idx == 1 or idx == 3 or idx == 5:
                for ele in elem.strip().split('#'):
                    if ele in dic:
                        tmp.append(dic[ele])
                    else:
                        tmp.append(dic['UNK'])
            else:
                tmp.extend(elem.strip().split('#'))
            all.append(tmp)
        return all

    def create_usr_dict(self):
        '''
        Words in labels must be included in vocab
        Save these words in a file for human check and backup
        '''
        my_voc = []
        for root, dirs, files in os.walk(self.raw_data):
            for name in files:
                path = os.path.join(root, name)
                print ('Processing path:', path)
                data = xlrd.open_workbook(path)
                table = data.sheet_by_name(self.sheetname)
                rows = table.nrows
                for i in tqdm(range(rows)):
                    label = table.cell(i, 2).value
                    my_voc.extend(label.split(','))
                    my_voc.append(table.cell(i, 1).value)
                my_voc.append('UNK')
                with codecs.open(self.usr_dict, 'w', 'utf-8') as f:
                    for ele in set(my_voc):
                        f.write(ele + '\n')

        return

    def generator(self):
        '''
        Process texts in the original document.
        :return: All texts in specific domain and all its domain words
        '''
        token_set = defaultdict(list)
        my_all_words = []
        jieba.load_userdict(self.usr_dict)
        for root, dirs, files in os.walk(self.raw_data):
            for name in files:
                path = os.path.join(root, name)
                print ('Processing path:', path)
                data = xlrd.open_workbook(path)
                table = data.sheet_by_name(self.sheetname)
                rows = table.nrows
                cols = table.ncols
                print ('rows:{}, cols:{}'.format(rows, cols))
                for i in tqdm(range(rows)):
                    # print ('row_number', i)
                    tit = self.tokenization(table.cell(i, 0).value.strip())
                    tit.append('title')
                    title = '#'.join(tit)
                    tit_len = len(tit)

                    lbl = table.cell(i, 2).value.strip().split(',')
                    label = '#'.join(lbl)

                    cot = self.tokenization(table.cell(i, 5).value.strip())
                    cot.append('sent')
                    content = '#'.join(cot)
                    cot_len = len(cot)

                    my_all_words.extend(tit)
                    my_all_words.extend(cot)
                    # find all word position in document
                    indices_title = [i for i, x in enumerate(tit) if x in lbl]
                    idx_title = '#'.join(map(str,indices_title))

                    indices_content = [i for i, x in enumerate(cot) if x in lbl]
                    idx_content = '#'.join(map(str, indices_content))

                    token_set[table.cell(i, 1).value.strip()].append([title, tit_len, content, cot_len, label, idx_title, idx_content])

            return token_set, my_all_words

    def collect_train_domains(self):
        domain_set = []
        for root, dirs, files in os.walk(self.raw_data):
            for name in files:
                path = os.path.join(root, name)
                print ('Processing path:', path)
                data = xlrd.open_workbook(path)
                table = data.sheet_by_name('Sheet1')
                rows = table.nrows
                cols = table.ncols
                print ('rows:{}, cols:{}'.format(rows, cols))
                for i in tqdm(range(rows)):
                    # print ('row_number', i)

                    domain_set.append(table.cell(i, 1).value.strip())

            return set(domain_set)


    def generate_test(self):
        token_set = []
        jieba.load_userdict(self.usr_dict)
        labeled_list = []
        for root, dirs, files in os.walk(self.test_data):
            for name in files:
                path = os.path.join(root, name)
                print ('Processing path:', path)
                data = xlrd.open_workbook(path)
                table = data.sheet_by_name(self.sheetname) #MySheet
                rows = table.nrows
                cols = table.ncols
                print ('rows:{}, cols:{}'.format(rows, cols))
                for i in tqdm(range(25, rows)):
                    # print ('row_number', i)
                    if table.cell(i, 2).value.strip() in self.domain_set:
                        domain = table.cell(i, 2).value.strip()
                        tit = self.tokenization(table.cell(i, 1).value.strip())
                        tit.append('title')
                        title = '#'.join(tit)
                        tit_len = len(tit)

                        cot = self.tokenization(table.cell(i, 5).value.strip())
                        cot.append('sent')
                        content = '#'.join(cot)
                        cot_len = len(cot)

                        # find all word position in document
                        token_set.append(
                            [domain, title, tit_len, content, cot_len])

                        labeled_list.append(i)

        tmp_label = os.path.join(self.data_dir, 'tmp_label')
        with open(tmp_label, 'wb') as f:
            pickle.dump(labeled_list, f)

        return token_set

    def write2csv(self, token_set, vocab, train=True):
        '''
        This func writes to some csv for training
        :param token_set: all processed words
        :param vocab: the specific vocab in some domain
        '''
        if train:
            with open(self.train_out, 'w') as f, open(self.dev_out, 'w') as f_dev:
                output_train = csv.writer(f)
                output_dev = csv.writer(f_dev)
                output_train.writerow(
                    ['class', 'titles', 'tit_len', 'content', 'cnt_len', 'labels', 'indices_title', 'indices_content'])
                output_dev.writerow(
                    ['class', 'titles', 'tit_len', 'content', 'cnt_len', 'labels', 'indices_title', 'indices_content'])
                for key, val in token_set.items():
                    tot_len = len(val)
                    # print ('Total train instance:', int(tot_len * 0.85))
                    # print ('Total dev instance:', tot_len - int(tot_len * 0.85))
                    for idx, ele in enumerate(val):
                        ele.insert(0, key)
                        if idx < int(tot_len * 0.85):
                            output_train.writerow([','.join(map(str, ele)) for ele in self.to_idx(ele, vocab)])
                        else:
                            output_dev.writerow([','.join(map(str, ele)) for ele in self.to_idx(ele, vocab)])
        else:
            with open(self.test_out, 'w') as f_test:
                output_test = csv.writer(f_test)
                output_test.writerow(['class', 'titles', 'tit_len', 'content', 'cnt_len'])
                for val in token_set:
                    for idx, ele in enumerate(val):
                        output_test.writerow([','.join(map(str, ele)) for ele in self.to_idx(ele, vocab)])
            return


if __name__ == '__main__':
    Data_extraction('raw_data', 'MySheet', './usr.dict', './stop_words.txt', 'all', 'test_raw', True)