# -*- coding: utf-8 -*-
"""
@Time  : 18/7/3 下午7:39 

@author: Samzhanshi
This class is to provide data for each batch
To add stochasticity
1. It splits into different buckets
2. It can be shuffled reaching the end
"""
import numpy as np

MAX_LEN = 4096

class BucketedDataIterator():
    def __init__(self, df, num_buckets=5):
        self.df = df
        self.total = len(df)
        df_sort = df.sort_values('cnt_len').reset_index(drop=True)
        self.size = self.total / num_buckets
        self.dfs = []
        for bucket in range(num_buckets - 1):
            self.dfs.append(df_sort.ix[bucket*self.size: (bucket + 1)*self.size - 1])
        self.dfs.append(df_sort.ix[(num_buckets-1)*self.size:])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.pos = 0
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        '''
        sorts dataframe by sequence length, but keeps it random within the same length
        '''
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, batch_size):
        '''
        :param batch_size: batch_size
        :return: A batch information for feeding
        csv structure class,titles,tit_len,content,cnt_len,labels,indices_title,indices_content
        '''
        if np.any(self.cursor + batch_size + 1 > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0, self.num_buckets)

        res = self.dfs[i].ix[self.cursor[i]:self.cursor[i] + batch_size - 1]

        titles = map(lambda x: map(int, x.split(",")), res['titles'].tolist())
        contents = map(lambda x: map(int, x.split(",")), res['content'].tolist())
        category = np.array(res['class'].tolist())
        # print (res['indices_title'].tolist())
        idx_tit = map(lambda x: map(int, x.split(",") if isinstance(x, str) else []), res['indices_title'].tolist())
        idx_cot = map(lambda x: map(int, x.split(",") if isinstance(x, str) else []), res['indices_content'].tolist())

        self.cursor[i] += batch_size

        # Pad sequences with 0s so they are all the same length
        max_tit_len = max(res['tit_len'])
        max_con_len = min(max(res['cnt_len']), MAX_LEN)
        tit_x = np.zeros([batch_size, max_tit_len], dtype=np.int32)
        tit_y = np.zeros([batch_size, max_tit_len], dtype=np.int32)
        for i, x_i in enumerate(tit_x):
            x_i[:res['tit_len'].values[i]] = titles[i]
        for i, x_i in enumerate(tit_y):
            for ele in idx_tit[i]:
                x_i[ele] = 1
        con_x = np.zeros([batch_size, max_con_len], dtype=np.int32)
        con_y = np.zeros([batch_size, max_con_len], dtype=np.int32)
        for i, y_i in enumerate(con_x):
            y_i[:min(res['cnt_len'].values[i], max_con_len)] = contents[i][:max_con_len]
        for i, x_i in enumerate(con_y):
            for ele in idx_cot[i]:
                if ele < max_con_len:
                    x_i[ele] = 1

        return category, tit_x, con_x, res['tit_len'].values, [min(i, max_con_len) for i in res['cnt_len'].values], tit_y, con_y

    def next_all_batch(self, batch_size, test=False):
        '''
        :param batch_size: batch_size
        :return: A batch information for feeding without changing any order
        '''
        res = self.df.ix[self.pos : self.pos + batch_size - 1]
        titles = map(lambda x: map(int, x.split(",")), res['titles'].tolist())
        contents = map(lambda x: map(int, x.split(",")), res['content'].tolist())
        category = np.array(res['class'].tolist())

        self.pos += batch_size
        # when reaching end, go to the begining.
        if self.pos == self.total:
            self.pos = 0
        max_tit_len = max(res['tit_len'])
        max_con_len = min(max(res['cnt_len']), MAX_LEN)
        tit_x = np.zeros([batch_size, max_tit_len], dtype=np.int32)
        tit_y = np.zeros([batch_size, max_tit_len], dtype=np.int32)
        for i, x_i in enumerate(tit_x):
            x_i[:res['tit_len'].values[i]] = titles[i]
        # label each position in the sentence

        con_x = np.zeros([batch_size, max_con_len], dtype=np.int32)
        con_y = np.zeros([batch_size, max_con_len], dtype=np.int32)
        for i, y_i in enumerate(con_x):
            y_i[:min(res['cnt_len'].values[i], max_con_len)] = contents[i][:max_con_len]
        # label each position in the sentence with max size
        if test is False:
            idx_tit = map(lambda x: map(int, x.split(",") if isinstance(x, str) else []), res['indices_title'].tolist())
            idx_cot = map(lambda x: map(int, x.split(",") if isinstance(x, str) else []),
                          res['indices_content'].tolist())

            for i, x_i in enumerate(tit_y):
                for ele in idx_tit[i]:
                    x_i[ele] = 1
            for i, x_i in enumerate(con_y):
                for ele in idx_cot[i]:
                    if ele < max_con_len:
                        x_i[ele] = 1
            return category, tit_x, con_x, res['tit_len'].values, [min(i, max_con_len) for i in res['cnt_len'].values], tit_y, con_y
        else:
            return category, tit_x, con_x, res['tit_len'].values, [min(i, max_con_len) for i in res['cnt_len'].values]

    def print_info(self):
        print ('dfs shape: ', [len(self.dfs[i]) for i in range(len(self.dfs))])
        print ('size: ', self.size)

    def get_answer(self):
        '''
        :return: return the answer
        '''
        pos_lb = map(lambda x: map(int, x.split(",")), self.df['labels'].tolist())
        return pos_lb

    def get_src(self):
        '''
        :return: source sentences for labeling
        '''
        title = map(lambda x: map(int, x.split(",")), self.df['titles'].tolist())
        content = map(lambda x: map(int, x.split(",")), self.df['content'].tolist())
        cont = [line[:MAX_LEN] for line in content]
        return (title, cont)

