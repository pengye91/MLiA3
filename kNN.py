#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: xyp
from typing import List, Dict

from numpy import *
import operator

from numpy.core.multiarray import zeros


def create_data_set() -> (ndarray, List[str]):
    group: ndarray = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels: List[str] = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(input_x: ndarray, data_set: ndarray, labels: ndarray, k: int) -> str:
    """
    :param input_x: 需要被分类的条目的特征值向量，(1, n)
    :param data_set: 训练样本， (m, n)， m条样本
    :param labels: 标签向量, (m, 1)
    :param k: 最邻近的数目
    :return: 输入条目类别
    """
    data_set_size: int = data_set.shape[0]  # 样本集合的条目数
    # 为了计算欧氏距离，先构造输入条目的矩阵和样本集合的距离差矩阵
    diff_mat: ndarray = tile(input_x, (data_set_size, 1)) - data_set
    sq_diff_mat: ndarray = diff_mat ** 2
    # 计算和样本集合里每条样本的距离，构成一个1 * m的向量
    distances: ndarray = sq_diff_mat.sum(axis=1) ** 0.5
    sorted_distances_indices: ndarray = distances.argsort()
    # 计算距离最短的前k个样本的类别个数
    class_count: Dict[str, int] = {}
    for i in range(k):
        sample_label = labels[sorted_distances_indices[i]]
        class_count[sample_label] = class_count.get(sample_label, 0) + 1
    # operator.itemgetter(1)返回一个callable，功能类似getitem.
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename: str) -> (ndarray, List[int]):
    with open(filename, 'r') as fr:
        lines = fr.readlines()
        number_of_lines: int = len(lines)
        return_matrix: ndarray = zeros((number_of_lines, 3))
        class_label_vector: List[int] = []
        index = 0
        for line in lines:
            line = line.strip()
            return_matrix[index, :] = line.split('\t')[:3]
            class_label_vector.append(int(line.split('\t')[3]))
            index += 1
    return return_matrix, class_label_vector


def auto_norm(data_set: ndarray) -> (ndarray, ndarray, ndarray):
    # 每列的最小值
    min_values: ndarray = data_set.min(axis=0)
    max_values: ndarray = data_set.max(axis=0)
    ranges: ndarray = max_values - min_values
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_values, (m, 1))
    # 在numpy中， /表示值相除，而不是矩阵相除
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_values


def dating_class_test():
    # not sure what a ho_ratio is.
    ho_ratio = 0.1
    dating_data_mat, dating_labels = file2matrix('./MLiA/Ch02/datingTestSet2.txt')
    norm_matrix, ranges, min_values = auto_norm(dating_data_mat)
    m = norm_matrix.shape[0]
    num_test_vecs = int(m*ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classfier_result = classify0(
            input_x=norm_matrix[i, :],
            data_set=norm_matrix[num_test_vecs:m, :],
            labels=dating_labels[num_test_vecs:m],
            k=3
        )
        print("the classfier result is %s, the real anwser is %d" % (classfier_result, dating_labels[i]))
        if classfier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: %f" % (error_count/float(num_test_vecs)))



