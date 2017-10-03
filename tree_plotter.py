#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: xyp
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

plt.rcParams['font.sans-serif'] = 'WenQuanYi Micro Hei'

decision_node: Dict = dict(boxstyle='sawtooth', fc='0.8')
leaf_node: Dict = dict(boxstyle='round4', fc='0.8')
arrow_args: Dict = dict(arrowstyle='<-')


def create_plot(input_tree: Dict):
    fig: Figure = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    plot_tree.totalW = float(get_leaf_number(input_tree))
    plot_tree.totalD = float(get_tree_depth(input_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(input_tree, (0.5, 1.0), '')
    plt.show()


def plot_node(node_txt, center_point, parent_point, node_type):
    create_plot.ax1.annotate(
        node_txt,
        xy=parent_point,
        xycoords='axes fraction',
        xytext=center_point,
        textcoords='axes fraction',
        va='center',
        ha='center',
        bbox=node_type,
        arrowprops=arrow_args
    )


def get_leaf_number(tree: Dict) -> int:
    leaf_number = 0
    first_key = list(tree.keys())[0]
    second_dict: Dict = tree[first_key]
    for k in second_dict.keys():
        if type(second_dict[k]).__name__ == 'dict':
            leaf_number += get_leaf_number(second_dict[k])
        else:
            leaf_number += 1
    return leaf_number


def get_tree_depth(tree: Dict) -> int:
    max_depth = 0
    first_key = list(tree.keys())[0]
    second_dict: Dict = tree[first_key]
    for k in second_dict.keys():
        if type(second_dict[k]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[k])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                     ]
    return list_of_trees[i]


def plot_mid_text(counter_point, parent_point, txt_string):
    x_mid = (parent_point[0] - counter_point[0]) / 2.0 + counter_point[0]
    y_mid = (parent_point[1] - counter_point[1]) / 2.0 + counter_point[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(tree: Dict, parent_point, node_txt):
    leaf_number: int = get_leaf_number(tree)
    depth: int = get_tree_depth(tree)
    first_key = list(tree.keys())[0]
    counter_point = (plot_tree.xOff + (1.0 + float(leaf_number)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(counter_point, parent_point, node_txt)
    plot_node(first_key, counter_point, parent_point, decision_node)
    second_dict = tree[first_key]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], counter_point, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), counter_point, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), counter_point, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD
