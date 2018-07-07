# -*- coding: utf-8 -*-
"""
@Time  : 18/7/3 下午7:39

@author: Samzhanshi
This file is for evalution
"""
import numpy as np
import pickle as p
from Config import Config
config = Config()

def evaluate(pred_prob, src, answer, bar=1.0):
    title_prob, cont_prob = pred_prob
    title, content = src
    # for t_p, c_p, t_l, c_l in zip(title_prob, cont_prob, title_lens, content_lens):
    # bz * seq
    tit_pred = [np.argmax(arr * np.array([1.0, bar]), 2).reshape(-1) for arr in title_prob]
    # print ('tit_pred', len(tit_pred))
    # print ('tit_shape', (tit_pred[0]).shape)
    cont_pred = [np.argmax(arr * np.array([1.0, bar]), 2).reshape(-1) for arr in cont_prob]
    res = []
    correct = 0
    for t_p, c_p, t_n, c_n, a_s in zip(tit_pred, cont_pred, title, content, answer):
        res_p = []
        min_l = min(len(t_p), len(t_n))
        for ele_p, ele_n in zip(t_p[:min_l], t_n[:min_l]):
            # print ('ele_p', ele_p)
            if ele_p == 1:
                res_p.append(ele_n)
        min_l = min(len(c_p), len(c_n))
        for ele_p, ele_n in zip(c_p[:min_l], c_n[:min_l]):
            if ele_p == 1:
                res_p.append(ele_n)
        for ele in set(res_p):
            if ele in a_s:
                correct += 1
        res.append(res_p)

    tot_pred = sum(len(set(x)) for x in res)
    tot_recall = sum(len(x) for x in answer)
    tot_res = [set(x) for x in res]
    assert len(tot_res) == len(answer)

    # with open('./data_entertain/dict','rb') as f:
    #     text = p.load(f)
    # for i in range(len(tot_res)):
    #     print ('pred:')
    #     print (','.join([text[m] for m in tot_res[i]]).encode('utf-8'))
    #     print ('real:')
    #     print (','.join([text[m] for m in answer[i]]).encode('utf-8'))

    if correct == 0:
        return 0.0, 0.0, 0.0

    prec, recall = correct * 1.0 / tot_pred, correct * 1.0 / tot_recall
    f_val = 2 * prec * recall / (prec + recall)

    # print ('answers', [set(x) for x in res])

    return prec, recall, f_val

def take_out(pred_prob, src, bar=1.0):
    title_prob, cont_prob = pred_prob
    title, content = src
    # for t_p, c_p, t_l, c_l in zip(title_prob, cont_prob, title_lens, content_lens):
    # bz * seq
    tit_pred = [np.argmax(arr * np.array([1.0, bar]), 2).reshape(-1) for arr in title_prob]
    # print ('tit_pred', len(tit_pred))
    # print ('tit_shape', (tit_pred[0]).shape)
    cont_pred = [np.argmax(arr * np.array([1.0, bar]), 2).reshape(-1) for arr in cont_prob]
    res = []
    for t_p, c_p, t_n, c_n in zip(tit_pred, cont_pred, title, content):
        res_p = []
        min_l = min(len(t_p), len(t_n))
        for ele_p, ele_n in zip(t_p[:min_l], t_n[:min_l]):
            # print ('ele_p', ele_p)
            if ele_p == 1:
                res_p.append(ele_n)
        min_l = min(len(c_p), len(c_n))
        for ele_p, ele_n in zip(c_p[:min_l], c_n[:min_l]):
            if ele_p == 1:
                res_p.append(ele_n)
        res.append(res_p)

    tot_res = [set(x) for x in res]

    with open(config.dictPath,'rb') as f, open(config.testRows,'rb') as ff:
        text = p.load(f)
        ids = p.load(ff)
        # print (ids)
    for idx, (res, i) in enumerate(zip(tot_res, ids)):
        print ('pred:', i)
        print (','.join([text[m] for m in tot_res[idx]]).encode('utf-8'))


if __name__ == '__main__':
    a = np.array([[[0.8, 0.2],[0.4, 0.6]],[[0.1, 0.9],[0.4, 0.6]]])
    b = np.array([[[0.8, 0.2],[0.4, 0.6]],[[0.1, 0.9],[0.4, 0.6]]])
    src_a = np.array([[10, 19], [19]])
    src_b = np.array([[20, 12], [11]])
    ans = np.array([[19], [11]])
    print (evaluate((a,b),(src_a,src_b),ans))


