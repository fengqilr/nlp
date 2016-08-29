__author__ = 'lian'
# -*- coding: utf-8 -*-

import pandas as pd
import logging
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import svds
import MySQLdb
import os
import gc
import matplotlib.pyplot as plt
import networkx as net
import sklearn.decomposition.truncated_svd as t_svd
from sklearn.metrics import *
from sklearn.cluster import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from model_projects.lianrui_model.dialysis.wordmaker.wordmaker import *
import jieba
from gensim import models, corpora
from gensim.models.doc2vec import *
from model_projects.lianrui_model.tool.utility import *
import re
from urllib import urlopen
from bs4 import BeautifulSoup
from scipy.sparse import *
from sklearn.externals import joblib

def is_qq_func(num_str):
    """
    确定是否是qq号码
    :param num_str:
    :return:
    """
    qq_zone_url = "http://user.qzone.qq.com/%s" % num_str
    try:
        f = urlopen(qq_zone_url)
    except Exception, e:
        print str(e)
    soup = BeautifulSoup(f.read(), "lxml")
    f.close()
    title_content = soup.find("title")
    if title_content.string == "404":
        is_qq = False
    else:
        is_qq = True
    return is_qq


def clean_sentence(raw_sentence):
    """
    标准化中文
    :param raw_sentence:
    :return:
    """
    pattern = re.compile(u'([\u4e00-\u9fa5\w]+)')
    return "".join(pattern.findall(raw_sentence))


def split_paragraph_2_sentence(paragraph):
    """
    对段落分句，并清理数据
    :param paragraph:
    :return:
    """
    split_pattern = re.compile(ur"[,?。，？!！]+")
    raw_sentence_list = split_pattern.split(paragraph)
    cleaned_sentence_list = map(lambda x: clean_sentence(x), raw_sentence_list)
    return cleaned_sentence_list

def transfer_doc_2_word_vec(word_2_vec_model, word_list, feature_word_dict={}):
    if len(word_list) == 0:
        doc_vec = np.float32(np.zeros(word_2_vec_model.vector_size))
    else:
        word_s = pd.Series(word_list)
        if feature_word_dict != {}:
            is_feature = word_s.map(lambda x: feature_word_dict.get(x, 0))
            word_s = word_s[is_feature==1]
        word_vec_s = word_s.map(lambda x: word_2_vec_model[x] if word_2_vec_model.vocab.has_key(x) else np.zeros(word_2_vec_model.vector_size).astype(np.float32))
        doc_vec = word_vec_s.values.sum(axis=0)
    return doc_vec


def transfer_doc_2_word_vec_v2(word_2_vec_dict, wor_vec_len, tfidf_clf, sent_list):
    sent_tfidf = tfidf_clf.transform(sent_list)
    word_list = tfidf_clf.vocabulary
    default_arr = np.zeros(wor_vec_len)
    word_2_vec_arr = np.array([word_2_vec_dict.get(word, default_arr) for word in word_list])
    sent_word_vec = sent_tfidf * word_2_vec_arr
    return sent_word_vec

def std_feat(feat_series):
    feat_max = feat_series.max()
    feat_min = feat_series.min()
    std_feat = feat_series.map(lambda x: (x-feat_min)/(feat_max - feat_min))
    return std_feat

agent_file_path = u"E:\数据资料\建模\透析\中介"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def transfer_sent_2_word_sim_vec(sent_word_info_dict, word_item_list, word_sim_dict):
    if len(sent_word_info_dict) == 0:
        word_sim_arr = np.zeros(len(word_item_list))
    else:
        word_item_s = pd.Series(word_item_list)
        # logging.info('stage 2')
        sent_word_list = sent_word_info_dict.keys()
        # logging.info('stage 3')
        word_item_sim_arr = word_item_s.map(lambda word: np.array([word_sim_dict.get((sent_word, word), 0) for sent_word in sent_word_list]))
        # logging.info('stage 4')
        word_item_sim = word_item_sim_arr.map(lambda x: x.max()*sent_word_info_dict[sent_word_list[x.argmax()]])
        word_sim_arr = word_item_sim.values
    return word_sim_arr


from itertools import product
def transfer_sent_2_word_sim_vec_v2(sent_word_info_dict, word_item_list, word_sim_dict):
    """
    根据输入sentence word info 和 词典 vec，生成word sim vec
    :param sent_word_info_dict:
    :param word_item_list:
    :param word_sim_dict:
    :return:
    """
    logging.info('stage 1')
    sent_word_list = sent_word_info_dict.keys()
    logging.info('stage 2')
    sent_word_feature_word_pair_iter = product(sent_word_list, word_item_list)
    logging.info('stage 3')
    sent_word_feature_word_pair_info = map(lambda word_pair: (word_pair[0], word_pair[1], word_sim_dict.get((word_pair[0], word_pair[1]), 0)), sent_word_feature_word_pair_iter)
    logging.info('stage 4')
    sent_word_feature_word_pair_df = pd.DataFrame(sent_word_feature_word_pair_info, columns=['sent_word', 'feature_word', 'cos_sim'])
    logging.info('stage 5')
    sent_word_feature_word_pair_df.sort_values(by=["feature_word", "cos_sim"], inplace=True)
    logging.info('stage 6')
    sent_word_feature_word_pair_df.drop_duplicates(subset=['feature_word'], inplace=True, keep='last')
    logging.info('stage 7')
    sent_word_feature_word_pair_df['sent_word_info'] = sent_word_feature_word_pair_df['sent_word'].map(lambda x: sent_word_info_dict[x])
    logging.info('stage 8')
    sent_word_feature_word_sim = sent_word_feature_word_pair_df['sent_word_info'] * sent_word_feature_word_pair_df['cos_sim']
    logging.info('stage 9')
    sent_word_feature_word_sim_dict = dict(zip(sent_word_feature_word_pair_df['feature_word'].values, sent_word_feature_word_sim.values))
    return sent_word_feature_word_sim_dict


def transfer_sent_2_word_sim_vec_v3(sent_word_info_dict, feature_word_item_list, word_sim_dict):
    feature_word_item_dict = {word: 1 for word in feature_word_item_list}
    sent_word_info_dict = {word: sent_word_info_dict[word] for word in sent_word_info_dict.keys() if feature_word_item_dict.get(word, -1) != -1}
    if len(sent_word_info_dict) == 0:
        word_sim_arr = np.zeros(len(feature_word_item_list))
    else:
        word_item_s = pd.Series(feature_word_item_list)
        # logging.info('stage 2')
        sent_word_list = sent_word_info_dict.keys()
        # logging.info('stage 3')
        word_item_sim_arr = word_item_s.map(lambda word: np.array([word_sim_dict.get(sent_word, {}).get(word, 0) if sent_word is not word else 1 for sent_word in sent_word_list]))
        # logging.info('stage 4')
        word_item_sim = word_item_sim_arr.map(lambda x: x.max()*sent_word_info_dict[sent_word_list[x.argmax()]])
        word_sim_arr = word_item_sim.values
    return word_sim_arr


def transfer_sent_2_word_sim_vec_v4(sent_word_info_dict, word_id_dict, word_sim_matrix):
    logging.info('stage 1')
    word_id_s = pd.Series(word_id_dict.values())
    logging.info('stage 2')
    sent_word_id_info_dict = {word_id_dict.get(word, -1): sent_word_info_dict[word] for word in sent_word_info_dict.keys() if word_id_dict.get(word, -1) != -1}
    sent_word_id_list = sent_word_id_info_dict.keys()
    logging.info('stage 3')
    word_item_sim_arr = word_id_s.map(lambda word_id: np.array([word_sim_matrix.getrow(sent_word_id).getcol(word_id).data if word_sim_matrix.getrow(sent_word_id).getcol(word_id).data.shape[0] == 1 else 0 for sent_word_id in sent_word_id_list]))
    logging.info('stage 4')
    word_item_sim = word_item_sim_arr.map(lambda x: x.max()*sent_word_id_info_dict[sent_word_id_list[x.argmax()]])
    return word_item_sim.values

def transfer_sent_2_word_sim_vec_v5(sent_word_info_dict, word_id_dict, word_sim_matrix):
    logging.info('stage 1')
    word_id_s = pd.Series(word_id_dict.values())
    logging.info('stage 2')
    sent_word_id_info_dict = {word_id_dict.get(word, -1): sent_word_info_dict[word] for word in sent_word_info_dict.keys() if word_id_dict.get(word, -1) != -1}
    sent_word_id_list = sent_word_id_info_dict.keys()
    logging.info('stage 3')
    word_item_sim_arr = word_id_s.map(lambda word_id: np.array([word_sim_matrix.getrow(sent_word_id).getcol(word_id).data if word_sim_matrix.getrow(sent_word_id).getcol(word_id).data.shape[0] == 1 else 0 for sent_word_id in sent_word_id_list]))
    logging.info('stage 4')
    word_item_sim = word_item_sim_arr.map(lambda x: x.max()*sent_word_id_info_dict[sent_word_id_list[x.argmax()]])
    return word_item_sim.values

def calculate_word_sim(feature_sim_info_df, sent_word_info_dict):
    feature_sim_info_df['sent_word_info'] = feature_sim_info_df['word'].map(lambda x: sent_word_info_dict.get(x, 0))
    if feature_sim_info_df['sent_word_info'].sum() == 0:
        return 0
    else:
        sent_feature_sim_info = feature_sim_info_df[feature_sim_info_df['sent_word_info'] > 0]
        sent_feature_sim_info['sent_word_sim'] = sent_feature_sim_info['cos_sim']*feature_sim_info_df['sent_word_info']
        return sent_feature_sim_info['sent_word_sim'].iloc[0]


def transfer_sent_2_word_sim_vec_v6(sent_word_info_dict, feature_word_item_list, word_sim_df):
    feature_word_item_dict = {word: 1 for word in feature_word_item_list}
    sent_word_info_dict = {word: sent_word_info_dict[word] for word in sent_word_info_dict.keys() if feature_word_item_dict.get(word, -1) != -1}
    if len(sent_word_info_dict) == 0:
        word_sim_arr = np.zeros(len(feature_word_item_list))
    else:
        word_sim_df['sim_info'] = word_sim_df['word_sim'].map(lambda x: calculate_word_sim(x, sent_word_info_dict))
        word_sim_arr = word_sim_df['sim_info'].values
    return word_sim_arr


def transfer_sent_2_word_sim_vec_v7(sent_word_info_dict, feature_word_item_list, word_sim_dict_list):
    # logging.info('stage 1')
    feature_word_item_dict = {word: 1 for word in feature_word_item_list}
    # logging.info('stage 2')
    sent_word_info_dict = {word: sent_word_info_dict[word] for word in sent_word_info_dict.keys() if feature_word_item_dict.get(word, 0) != 0}
    # logging.info('stage 3')
    sent_word_list = sent_word_info_dict.keys()
    sent_info_list = sent_word_info_dict.values()
    # logging.info('stage 4')
    feature_word_sent_word_sim_list = map(lambda x: np.array([x.get(word, 0) for word in sent_word_list]), word_sim_dict_list)
    # logging.info('stage 5')
    feature_word_sent_word_sim = [float(feature_sent_sim.max()*sent_info_list[feature_sent_sim.argmax()]) if len(feature_sent_sim) > 0 else 0 for feature_sent_sim in feature_word_sent_word_sim_list]
    return feature_word_sent_word_sim

def transfer_sent_2_word_sim_vec_v8(sent_word_info_dict, feature_word_item_list, word_sim_dict_list):
    # logging.info('stage 1')
    feature_word_item_dict = {word: 1 for word in feature_word_item_list}
    # logging.info('stage 2')
    sent_word_info_dict = {word: sent_word_info_dict[word] for word in sent_word_info_dict.keys() if feature_word_item_dict.get(word, 0) != 0}
    # logging.info('stage 3')
    sent_word_list = sent_word_info_dict.keys()
    sent_info_list = sent_word_info_dict.values()
    # logging.info('stage 4')
    feature_word_sent_word_sim_list = [np.array([word_sim_dict.get(word, 0) for word in sent_word_list]) for word_sim_dict in word_sim_dict_list]
    # logging.info('stage 5')
    feature_word_sent_word_sim = [feature_sent_sim.max()*sent_info_list[feature_sent_sim.argmax()] for feature_sent_sim in feature_word_sent_word_sim_list]
    return feature_word_sent_word_sim

def transfer_sent_2_word_sim_vec_v9(sent_word_info_dict, feature_word_item_list, word_sim_dict_list):
    # logging.info('stage 1')
    feature_word_item_dict = {word: 1 for word in feature_word_item_list}
    # logging.info('stage 2')
    sent_word_info_dict = {word: sent_word_info_dict[word] for word in sent_word_info_dict.keys() if feature_word_item_dict.get(word, 0) != 0}
    # logging.info('stage 3')
    sent_word_list = sent_word_info_dict.keys()
    sent_word_exist_dict = {word: 1 for word in sent_word_list}
    sent_info_list = sent_word_info_dict.values()
    # logging.info('stage 4')
    feature_word_sent_word_sim_list = [np.array([[sent_word_exist_dict.get(word, 0)*cos_sim, sent_word_info_dict.get(word, 0)] for word, cos_sim in word_sim_dict.items()]) for word_sim_dict in word_sim_dict_list]
    # logging.info('stage 5')
    feature_word_sent_word_sim = [float(feature_sent_sim[:, 0].max()*feature_sent_sim[:, 1][feature_sent_sim[:, 0].argmax()]) for feature_sent_sim in feature_word_sent_word_sim_list]
    return feature_word_sent_word_sim
