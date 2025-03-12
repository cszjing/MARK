import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import os
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import copy
from scipy.spatial.distance import pdist, squareform, cdist

def QbE(args, word_label, word_predict_phoc_label, epoch,
        query_index, fold=None, drop_first=True):
    start = datetime.now()
    if word_predict_phoc_label.shape[0] != len(word_label):
        raise ValueError('The number of word_phoc_label vectors and number of word_label must match')
    avg_precs = np.array([])
    
    threads = []
    sub_query_index = [query_index[i::args.num_threads] for i in range(args.num_threads)]
    for i in range(args.num_threads):
        t = MyThread(muli_thread_calculate_map, args=(word_predict_phoc_label[sub_query_index[i]], word_predict_phoc_label, args.distance_metric, word_label, sub_query_index[i], 'QbE',))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
        avg_precs = np.concatenate((avg_precs, t.get_result()))
    mAP = np.mean(avg_precs[avg_precs > 0])

    # 输出
    # print("Total test cases is {}, valid test cases is {}\n".format(len(query_index), len(avg_precs[avg_precs > 0])))
    # print("Time taken to calculate mAP of QbE: ", datetime.now() - start)
    return mAP

def QbS(args, query_phoc, word_predict_phoc_label, query_labels, word_label,
        epoch, query_index, fold=None, drop_first=False):
    start = datetime.now()
    if query_phoc.shape[1] != word_predict_phoc_label.shape[1]:
        print(query_phoc.shape[1], word_predict_phoc_label.shape[1])
        raise ValueError('Shape mismatch')
    if query_phoc.shape[0] != len(query_labels):
        raise ValueError('The number of query feature vectors and query labels does not match')
    if word_predict_phoc_label.shape[0] != len(word_label):
        raise ValueError('The number of test feature vectors and test labels does not match')

    avg_precs = np.array([])
    threads = []
    sub_query_index = [query_index[i::args.num_threads] for i in range(args.num_threads)]
    for i in range(args.num_threads):
        t = MyThread(muli_thread_calculate_map, args=(query_phoc[sub_query_index[i]], word_predict_phoc_label, args.distance_metric, word_label, sub_query_index[i],'QbS',))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        avg_precs = np.concatenate((avg_precs, t.get_result()))
    mAP = np.mean(avg_precs)
    # print("Time taken to calculate mAP of QbS: ", datetime.now() - start)
    return mAP, avg_precs


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

def average_precision(ret_vec_relevance, gt_relevance_num=None):
    # np.cumsum累加，1,1,1,0,1
    #  累加后        1,2,3,3,4
    ret_vec_relevance = np.array(ret_vec_relevance)
    ret_vec_cumsum = np.cumsum(ret_vec_relevance, dtype=float)
    ret_vec_range = np.arange(1, len(ret_vec_relevance) + 1)
    ret_vec_precision = ret_vec_cumsum / ret_vec_range  #

    if gt_relevance_num is None:
        n_relevance = ret_vec_relevance.sum()
    else:
        n_relevance = gt_relevance_num

    if n_relevance > 0:  # 排除与查询相关图像只有一个，即查询本身
        ret_vec_ap = (ret_vec_precision * ret_vec_relevance).sum() / n_relevance
    else:
        ret_vec_ap = 0.0
    return ret_vec_ap

def muli_thread_calculate_map(query_phoc, word_predict_phoc_label, distance_metric, word_label, query_index, flag):
    dists = cdist(XA=query_phoc, XB=word_predict_phoc_label, metric=distance_metric)

    # 将dists中的元素按‘行’从小到大排列（越小越相似），提取其对应的index(索引)，然后输出到inds
    inds = np.argsort(dists, axis=1)
    # 修正np.argsort（）中，带有查询本身的检索结果，删除查询本身所在位置
    temp_inds = np.zeros((len(query_index), word_label.shape[0] - 1))
    if flag == 'QbE':
        for j, i in enumerate(query_index):
            temp = list(inds[j])
            temp_inds[j] = np.delete(inds[j], temp.index(i))
        inds = temp_inds.astype(np.int_)

    # 单词矩阵，所有label复制word_num行
    retr_mat = np.tile(word_label, (len(query_index), 1))
    row_selector = np.transpose(np.tile(np.arange(len(query_index)), (inds.shape[1], 1)))
    retr_mat = retr_mat[row_selector, inds]  # 得到检索结果的矩阵，按相似程度由大到小排列的，单词矩阵
    rel_matrix = retr_mat == np.atleast_2d(word_label[query_index]).T  # 每一行中，得到的相同的单词标为true，否则为false
    avg_precs = np.array([average_precision(row) for row in rel_matrix])
    return avg_precs
