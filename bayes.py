#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: xyp
import random

from numpy import ones, log, array


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])  # create empty set
    for document in data_set:
        vocab_set = vocab_set | set(document)  # union of the two sets
    return list(vocab_set)


def set_of_words_2_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return return_vec


def train_NB_0(train_matrix, train_category):
    """
    计算条件概率p(w1, w2,..., wn | ci)和p(c1), p(c0)
    这个只是训练数据的概率。
    :param train_matrix:
    :param train_category:
    :return:
    """
    train_doc_number, word_number = len(train_matrix), len(train_matrix[0])
    # p_abusive 就是p_1，属于类别1的文档的概率
    # 而p_0 = 1 - p_abusive
    p_abusive = sum(train_category) / float(train_doc_number)
    # p_i_num 统计词的分布;分子, 长度为词汇表的长度
    p_0_num, p_1_num = ones(word_number), ones(word_number)
    # p_i_denom 统计次数；分母
    p_0_denom, p_1_denom = 2.0, 2.0

    for i in range(train_doc_number):
        if train_category[i] == 1:
            p_1_num += train_matrix[i]
            p_1_denom += sum(train_matrix[i])
        else:
            p_0_num += train_matrix[i]
            p_0_denom += sum(train_matrix[i])
    p_1_vect = log(p_1_num / p_1_denom)
    p_0_vect = log(p_0_num / p_0_denom)

    return p_0_vect, p_1_vect, p_abusive


def classify_NB(vec_2_classify, p_0_vect, p_1_vect, p_c_1):
    p_1 = sum(vec_2_classify * p_1_vect) + log(p_c_1)
    p_0 = sum(vec_2_classify * p_0_vect) + log(1.0 - p_c_1)
    return 1 if p_1 > p_0 else 0


def test_NB(test_entry):
    posts, classes = load_data_set()
    vocab_list = create_vocab_list(posts)
    train_matrix = [set_of_words_2_vec(vocab_list, post) for post in posts]
    p_0_vect, p_1_vect, p_abusive = train_NB_0(array(train_matrix), array(classes))
    doc = array(set_of_words_2_vec(vocab_list, test_entry))
    print(test_entry, "classified as: ", classify_NB(doc, p_0_vect, p_1_vect, p_abusive))


def bag_of_words_2_vect_MN(vocab_list, input_set):
    return_vect = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vect[vocab_list.index(word)] += 1
    return return_vect


def text_parse(big_string):
    import re
    token_list = re.split(r'\W*', big_string)
    return [tok.lower() for tok in token_list if len(tok) > 2]


def spam_test():
    from os import listdir
    from os.path import isfile, join

    spam_email_path = '/home/xyp/anaconda3/envs/MLiA/src/MLiA/Ch04/email/spam/'
    ham_email_path = '/home/xyp/anaconda3/envs/MLiA/src/MLiA/Ch04/email/ham/'

    doc_list, class_list, full_test = [], [], []
    spam_email_list = [f for f in listdir(spam_email_path) if isfile(join(spam_email_path, f))]
    ham_email_list = [f for f in listdir(ham_email_path) if isfile(join(ham_email_path, f))]

    for spam_email in spam_email_list:
        with open(spam_email_path + spam_email, 'rb') as f:
            try:
                word_list = text_parse(f.read().decode('ascii'))
            except UnicodeDecodeError:
                print(spam_email)
                pass
        doc_list.append(word_list)
        full_test.extend(word_list)
        class_list.append(1)

    for ham_email in ham_email_list:
        with open(ham_email_path + ham_email, 'rb') as f:
            try:
                word_list = text_parse(f.read().decode('ascii'))
            except UnicodeDecodeError:
                print(ham_email)
                pass
        doc_list.append(word_list)
        full_test.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)

    test_set = random.sample(range(50), 10)
    training_set = set(range(50)) - set(test_set)
    training_matrix, training_classes = [], []
    for i in training_set:
        training_matrix.append(set_of_words_2_vec(vocab_list, doc_list[i]))
        training_classes.append(class_list[i])

    p_0_v, p_1_v, p_abusive = train_NB_0(training_matrix, training_classes)

    error_count = 0
    for test_index in test_set:
        doc = set_of_words_2_vec(vocab_list, doc_list[test_index])
        if classify_NB(doc, p_0_v, p_1_v, p_abusive) != class_list[test_index]:
            error_count += 1
    print('the error rate is ', float(error_count)/len(test_set))


