#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: xyp
from numpy import exp, mat, shape, ones


def load_data_set(filename: str='/home/xyp/anaconda3/envs/MLiA/src/MLiA/Ch05/testSet.txt'):
    data_matrix, label_matrix = [], []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_array = line.strip().split()
            # x0 = 1.0, x1 = line_array[0], x2 = line_array[1]
            data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
            label_matrix.append(int(line_array[-1]))
    return data_matrix, label_matrix


def sigmoid(input_x):
    return 1.0 / (1 + exp(-input_x))


def gradient_ascent(input_data_matrix, input_class_labels):
    # data_matrix: 100 * 3: [x0, x1, x2]
    # label_matrix: 100 * 1
    data_matrix, label_matrix = mat(input_data_matrix), mat(input_class_labels).transpose()
    m, n = shape(data_matrix)
    # weights: 3 * 1
    alpha, max_cycles, weights = 0.001, 500, ones((n, 1))
    for k in range(max_cycles):
        # h是一个列向量，个数等于样本个数
        # data_matrix * weights: 100 * 1
        # 相当于对 100 × 1 这个输入矩阵的每个值计算一个sigmoid输出，最后返回一个 100 × 1 的矩阵.
        # h 就是当前的f(w
        h = sigmoid(data_matrix * weights)
        # error 是当前迭代得到的误差向量: 100 * 1
        error = label_matrix - h
        weights = weights + alpha * data_matrix.transpose() * error
    return weights

