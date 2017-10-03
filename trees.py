#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: xyp
import operator
from math import log
from typing import Dict, List
from numpy import array, ndarray

from collections import defaultdict


def shannon_ent(data_set) -> float:
    entry_number: int = len(data_set)
    label_counts: Dict[str, int] = defaultdict(lambda: 0)
    for feature_vector in data_set:
        current_label = feature_vector[-1]
        label_counts[current_label] += 1
    shannon_entropy = 0.0
    for k in label_counts.keys():
        prob = float(label_counts[k]) / entry_number
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    """
    依据特征值将数据集切分
    :param data_set:数据集合
    :param axis: 特征值在特征向量中的位置
    :param value: 切分依据的特征值
    :return: 依据特征值切分后的数据集
    """
    splited_data_set = []
    for feature_vector in data_set:
        if feature_vector[axis] == value:
            # 去除掉feature_vector[axis]剩下的feature_vector
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            splited_data_set.append(reduced_feature_vector)
    return splited_data_set


def choose_best_feature_to_split(data_set):
    """
    遍历所有特征，根据每个特征值切分数据集，切分之后再计算切分后的熵，计算出相应的每个信息增益
    依据最小的信息增益返回对应的特征值。
    在函数中调用的数据需要满足一定的要求：
        1. 数据必须是一种有列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度；
        2. 数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。
    :param data_set:
    :return:
    """
    feature_number = len(data_set[0]) - 1
    base_entropy = shannon_ent(data_set)
    best_info_gain, best_feature = 0.0, -1
    for i in range(feature_number):
        feature_list = [feature[i] for feature in data_set]
        unique_values = set(feature_list)
        new_entropy = 0.0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            sub_prob = len(sub_data_set) / float(len(data_set))
            new_entropy += sub_prob * shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(class_list: List):
    """
    该函数使用分类名称的列表，创建一个计算类别出现频率的字典，并返回出现次数最多的分类名称。
    :param class_list:
    :return:
    """
    class_count: Dict[str, int] = defaultdict(lambda: 0)
    for vote in class_list:
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set: List, labels: List) -> Dict:
    """
    决策树构造函数，递归调用。
    该递归函数的终止条件有两个：
        1. 所有的类标签完全相同，则直接返回该类别；
        2. 所有特征值都已经被划分完毕，可是仍然不能将数据集划分为仅包含唯一类别的分组，则返回出现次数最多的类别。
    :param data_set: 需要被划分的数据集
    :param labels: 标签列表
    :return: 决策树字典
    """
    class_list = [feature[-1] for feature in data_set]
    # 如果这个数据集里的类别完全相同，则停止划分数据集
    if class_list.count(class_list[0]) == len(class_list):
        # 返回该类别
        return class_list[0]
    # 如果该数据集里的特征向量中只包含类别，也就是说遍历完了所有的特征值（但是类别并不完全相同）
    # 这时返回出现次数最多的类别.
    if len(data_set[0]) == 1:
        return majority_count(class_list)
    # 如果两种情况都不满足，则循环递归计算。
    # 根据最佳特征值将数据集循环划分。
    best_feature = choose_best_feature_to_split(data_set)
    best_label = labels[best_feature]
    tree = {best_label: {}}
    # del 这一步非常重要！递归过程中将已经判断过的标签删去。
    del labels[best_feature]
    feature_values = set([feature[best_feature] for feature in data_set])
    print(feature_values)
    for feature_value in feature_values:
        sub_labels = labels[:]
        tree[best_label][feature_value] = create_tree(
            split_data_set(data_set, best_feature, feature_value),
            sub_labels
        )
    return tree


def classify(input_tree: Dict, feature_labels: List, test_feature_vec: List):
    """
    给定一个test_feature_vec特征值向量，输出这个特征向量所属的类别标签。
    运作思路和构建决策树类似：
        比较测试的特征值向量与决策树上的数值，递归执行知道进入叶子节点。最后返回该叶子节点的类别标签。
    :param input_tree: 事前准备好的决策树。
    :param feature_labels: 特征值标签向量。
    :param test_feature_vec: 需要被测试的特征向量。
    :return:
    """
    first_key = list(input_tree.keys())[0]
    second_dict: Dict = input_tree[first_key]
    feature_index: int = feature_labels.index(first_key)
    label = ''
    for k in second_dict.keys():
        if test_feature_vec[feature_index] == k:
            if type(second_dict[k]).__name__ == 'dict':
                label = classify(second_dict[k], feature_labels, test_feature_vec)
            else:
                label = second_dict[k]
    return label
