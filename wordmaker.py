# -*- coding=utf-8 -*-
import re
import collections
import math
import pandas as pd
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
# modify from https://gist.github.com/lastland/3322018


def info_entropy(words):
    """
    计算熵
    :param words:
    :return:
    """
    result = 0 
    total = sum([val for _, val in words.iteritems()])
    for word, cnt in words.iteritems():
        if word == u"":
            p = float(1)/total
            result -= cnt * p * math.log(p)
        else:
            p = float(cnt) / total
            result -= p * math.log(p)
    return result


def cal_min_entropy(word_info):
    """
    计算左右熵最小值
    :param word_info:
    :return:
    """
    return min(word_info['left_entropy'], word_info['right_entropy'])


def cal_synthesized_index(word_info):
    """
    计算左右熵最小值
    :param word_info:
    :return:
    """
    return word_info['word_coagulation_v2'] * word_info['min_entropy']


def make_words(content, filename, max_word_len, entropy_threshold):
    sentences = re.split("\W+|[a-zA-Z0-9]+", content, 0, re.UNICODE)
    #sentences = re.split(ur"[a-zA-Z0-9^\u4E00-\u9FA5]+", content, 0, re.UNICODE)
    freq = collections.Counter()
    for sentence in sentences:
        if sentence:
            l = len(sentence)
            wl = min(l, max_word_len)
            for i in range(1, wl + 1): 
                for j in range(0, l - i + 1): 
                    freq[sentence[j:j + i]] += 1
    # for key in freq.keys():
    #     print key
    total = sum([val for _, val in freq.iteritems()])
    ps = collections.defaultdict(int)
    for word, val in freq.iteritems():
        ps[word] = float(val) / total
    print freq[u'套现']
    words = set()
    word_coagulation_list = []
    for word, word_p in ps.items():
        if len(word) > 1:
            p = 0
            for i in range(1, len(word)):
                t = ps[word[0:i]] * ps[word[i:]]
                p = max(p, t)
            if freq[word] >= 3 and word_p / p > 100:
                words.add(word)
            # if freq[word] >= 5:
            #     word_coagulation_list.append({'word': word, 'word_coagulation': word_p / p})
    # for key in words:
    #     print key
    # word_coagulation_df = pd.DataFrame(word_coagulation_list)
    # print word_coagulation_df.columns
    # word_coagulation_df.sort(columns='word_coagulation', inplace=True, axis=1, ascending=True)
    # print word_coagulation_df.head(100)
    final_words = set()
    for word in words:
        lf = rf = True
        left_words = collections.Counter()
        right_words = collections.Counter()
        pattern = re.compile(word.join(['.?', '.?']))
        for sentence in sentences:
            l = pattern.findall(sentence)
            # print type(l)
            if l:
                if l[0][0] != word[0]:
                    left_words[l[0][0]] += 1
                else:
                    lf = False
                if l[0][-1] != word[-1]:
                    right_words[l[0][-1]] += 1
                else:
                    rf = False
        left_info_entropy = info_entropy(left_words)
        right_info_entropy = info_entropy(right_words)
        if lf and len(left_words) > 0 and left_info_entropy < entropy_threshold:
            continue
        if rf and len(right_words) > 0 and right_info_entropy < entropy_threshold:
            continue
        # if word == u"套现":
        #     print left_info_entropy, right_info_entropy
        final_words.add(word)
    words_list = list(final_words)
    words_list.sort(cmp = lambda x, y: cmp(freq[y], freq[x]))
    
    final_freq = collections.Counter()
    file = open(filename, 'w')

    #for word,v in final_freq.iteritems():
    #for word in final_words:
    for word in words_list:
        v = freq[word]
        file.write("%s %d\n" % (word,v))
        final_freq[word] = v

    file.close()

    return final_freq


def make_words_v2(content, filename, max_word_len, entropy_threshold):
    """
    自己的分词版本，修改计算左右词熵的模块
    :param content:
    :param filename:
    :return:
    """
    sentences = re.split("\W+|[a-zA-Z0-9]+", content, 0, re.UNICODE)
    #sentences = re.split(ur"[a-zA-Z0-9^\u4E00-\u9FA5]+", content, 0, re.UNICODE)
    freq = collections.Counter()
    for sentence in sentences:
        if sentence:
            l = len(sentence)
            wl = min(l, max_word_len)
            for i in range(1, wl + 1):
                for j in range(0, l - i + 1):
                    freq[sentence[j:j + i]] += 1
    # for key in freq.keys():
    #     print key
    total = sum([val for _, val in freq.iteritems()])
    ps = collections.defaultdict(int)
    for word, val in freq.iteritems():
        ps[word] = float(val) / total

    words = set()
    word_coagulation_list = []
    for word, word_p in ps.items():
        if len(word) > 1:
            p = 0
            for i in range(1, len(word)):
                t = ps[word[0:i]] * ps[word[i:]]
                p = max(p, t)
            if freq[word] >= 3 and word_p / p > 100:
                words.add(word)
            if freq[word] >= 5:
                word_coagulation_list.append({'word': word, 'word_coagulation': word_p / p})
        else:
            pass
            # words.add(word)
    # for key in words:
    #     print key
    word_coagulation_df = pd.DataFrame(word_coagulation_list)
    word_coagulation_df.sort(columns='word_coagulation', inplace=True, axis=0, ascending=False)
    print word_coagulation_df.columns
    print word_coagulation_df.head(100)
    final_words = set()
    for word in words:
        left_words = collections.Counter()
        right_words = collections.Counter()
        pattern = re.compile(word.join(['.?', '.?']))
        for sentence in sentences:
            l = pattern.findall(sentence)
            for tgt_sample in l:
                left_right_list = tgt_sample.split(word)
                if len(left_right_list) == 2:
                    left_words[left_right_list[0]] += 1
                    right_words[left_right_list[1]] += 1
            # # print type(l)
            # if l:
            #     if l[0][0] != word[0]:
            #         left_words[l[0][0]] += 1
            #     else:
            #         lf = False
            #     if l[0][-1] != word[-1]:
            #         right_words[l[0][-1]] += 1
            #     else:
            #         rf = False
        left_info_entropy = info_entropy(left_words)
        right_info_entropy = info_entropy(right_words)
        if len(left_words) == 0 and u"" in left_words and len(right_words) == 0 and u"" in right_words:
            final_words.add(word)
        elif left_info_entropy > entropy_threshold and right_info_entropy > entropy_threshold:
            final_words.add(word)
        else:
            pass
    words_list = list(final_words)
    words_list.sort(cmp = lambda x, y: cmp(freq[y], freq[x]))

    final_freq = collections.Counter()
    file = open(filename, 'w')
    for word in words_list:
        v = freq[word]
        file.write("%s %d\n" % (word,v))
        final_freq[word] = v

    file.close()

    return final_freq


def make_words_v3(content, max_word_len=5, min_word_freq=10):
    """
    生成新词version3, 返回新词的频次:freq，内部凝固度: word_coagulation_v1, word_coagulation_v2,
    自由度（左熵，右熵，左右熵的最小值）: left_entropy, right_entropy, min_entropy,
    综合指标（左右熵最小值乘以内部凝固度v2 min_entropy * word_coagulation_v2）: synthesized_index
    :param content: 新词原始文本
    :param max_word_len: 新词最大长度
    :param min_word_freq: 新词最小频次
    :return:
    """
    sentences = re.split("\W+|[a-zA-Z0-9]+", content, 0, re.UNICODE)
    word_freq = collections.Counter()
    word_left_right_freq = collections.Counter()
    for sentence in sentences:
        if sentence:
            l = len(sentence)
            wl = min(l, max_word_len)
            for i in range(1, wl + 1):
                for j in range(0, l - i + 1):
                    word = sentence[j: j+i]
                    word_freq[word] += 1
                    left_word_index = j -1
                    right_word_index = j + i
                    if left_word_index < 0:
                        left_word = u""
                    else:
                        left_word = sentence[left_word_index]
                    if right_word_index >= l:
                        right_word = u""
                    else:
                        right_word = sentence[right_word_index]
                    if i > 1:
                        word_left_right_freq[(left_word, word)] += 1
                        word_left_right_freq[(word, right_word)] += 1
    # word_freq[u''] = 0
    total = sum([val for _, val in word_freq.iteritems()])
    ps = collections.defaultdict(int)
    for word, val in word_freq.iteritems():
        ps[word] = float(val) / total
    word_coagulation_list = []
    for word, word_p in ps.items():
        if len(word) > 1:
            p = 0
            for i in range(1, len(word)):
                t = ps[word[0:i]] * ps[word[i:]]
                p = max(p, t)
            if word_freq[word] >= min_word_freq:
                word_coagulation_list.append({'word': word, 'word_coagulation_v1': word_p/p,
                                              'word_coagulation_v2': word_p*word_p / p, 'freq': word_freq[word]})
    word_coagulation_df = pd.DataFrame(word_coagulation_list)
    word_left_right_freq_list = []
    for (lf_w, rh_w), val in word_left_right_freq.iteritems():
        if lf_w == u"" or rh_w == u"":
            for i in range(val):
                word_left_right_freq_list.append({"left_word": lf_w, 'right_word': rh_w, 'freq': 1, 'left_word_cnt': word_freq[lf_w], 'right_word_cnt': word_freq[rh_w]})
        else:
            word_left_right_freq_list.append({"left_word": lf_w, 'right_word': rh_w, 'freq': val, 'left_word_cnt': word_freq[lf_w], 'right_word_cnt': word_freq[rh_w]})
    word_left_right_freq_df = pd.DataFrame(word_left_right_freq_list)
    word_right_freq_df = word_left_right_freq_df[word_left_right_freq_df['left_word'] != u""]
    word_right_freq_df = word_right_freq_df[word_right_freq_df['left_word_cnt'] >= min_word_freq]
    word_right_freq_df['prob'] = word_right_freq_df['freq'].map(float).values/word_right_freq_df['left_word_cnt'].values
    word_right_freq_df['right_entropy'] = word_right_freq_df['prob'].map(lambda x: -x * math.log(x))
    word_right_entropy = word_right_freq_df.groupby(by='left_word')['right_entropy'].sum()
    word_left_freq_df = word_left_right_freq_df[word_left_right_freq_df['right_word'] != u""]
    word_left_freq_df = word_left_freq_df[word_left_freq_df['right_word_cnt'] >= min_word_freq ]
    word_left_freq_df['prob'] = word_left_freq_df['freq'].map(float).values/word_left_freq_df['right_word_cnt'].values
    word_left_freq_df['left_entropy'] = word_left_freq_df['prob'].map(lambda x: -x * math.log(x))
    word_left_entropy = word_left_freq_df.groupby(by='right_word')['left_entropy'].sum()
    word_right_entropy = word_right_entropy.reset_index()
    word_left_entropy = word_left_entropy.reset_index()
    word_right_entropy.columns = ['word', 'right_entropy']
    word_left_entropy.columns = ['word', 'left_entropy']
    word_left_right_entropy = pd.merge(word_right_entropy, word_left_entropy, how='inner', on='word')
    word_coagulation_left_right_info = pd.merge(word_coagulation_df, word_left_right_entropy, how='inner', on='word')
    word_coagulation_left_right_info['min_entropy'] = word_coagulation_left_right_info.apply(cal_min_entropy, axis=1)
    word_coagulation_left_right_info['synthesized_index'] = word_coagulation_left_right_info.apply(cal_synthesized_index, axis=1)
    word_coagulation_left_right_info.sort_values(by='synthesized_index', ascending=False, inplace=True)
    return word_coagulation_left_right_info

# import time
# use_time_list = []
# content_cnt_list = []
# import MySQLdb
# conn = MySQLdb.connect(host='192.168.200.122', port=3306, user='uatuser_dev', passwd='develop20151215', db='crawler_bigdata_dev')
# cur = conn.cursor()
# max_page = 5
# content_str_list = []
# for i in range(max_page):
#     sql_cmd = 'SELECT title FROM baidu_tieba_%02d' % i
#     cur.execute(sql_cmd)
#     rst = cur.fetchall()
#     title_df = pd.DataFrame(list(rst), columns=['title'])
#     title_df.dropna(inplace=True)
#     content_str_list.extend(list(title_df['title'].map(lambda x: x.replace(u"回复：", u"")).values))
# content_str_list = list(set(content_str_list))
# total_len = len(content_str_list)
# for i in range(1, 11):
#     content_str = u",".join(content_str_list[0: int(total_len*i/10.0)])
#     content_cnt_list.append(total_len*i/10.0)
#     start_time = time.clock()
#     make_words_v3(content_str)
#     use_time = time.clock() - start_time
#     use_time_list.append(use_time)
#     print total_len*i/10.0
#     print use_time
# conn.close()
