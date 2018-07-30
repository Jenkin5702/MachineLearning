# coding=utf-8
from scipy import log


# 信息熵
def entropy(data):
    group = data.groupby(data.columns[-1])
    m = data.shape[0] + .0
    ent = 0
    for name, g in group:
        p = g.shape[0] / m
        if p != 0:
            ent = ent - p * log(p) / log(2)
    return ent


# 信息增益
def information_gain(data, attr):
    gain = entropy(data)
    m = data.shape[0] + .0
    group = data.groupby(attr)
    for name, g in group:
        v = g.shape[0] / m
        gain = gain - v * entropy(g)
    return gain


# 增益率
def gain_ratio(data, attr):
    iv = 0
    m = data.shape[0] + .0
    group = data.groupby(attr)
    for name, g in group:
        r = g.shape[0] / m
        r = r * log(r) / log(2)
        iv = iv - r
    return information_gain(data, attr) / iv


# 基尼指数
def gini(data):
    gn = 1
    m = data.shape[0] + .0
    group = data.groupby(data.columns[-1])
    for name, g in group:
        r = g.shape[0] / m
        gn = gn - r ** 2
    return gn


def gini_index(data, attr):
    gni = 0
    m = data.shape[0] + .0
    group = data.groupby(attr)
    for name, g in group:
        p = g.shape[0] / m
        gni = gni + p * gini(data)
    return gni

