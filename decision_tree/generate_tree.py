# coding=utf-8
import pandas as pd
from numpy import array
from attribute_select import *

ml = pd.read_excel("西瓜数据集.xlsx").drop(['bianhao', 'hantang', 'midu'], axis=1)


# print(ml)


def generate_tree(data):
    node = {'leaf': True, 'dat': array(data.index),'branch':{}}
    group = data.groupby(data.columns[-1])
    if group.ngroups == 1:
        node['type'] = group.groups.keys()[0]
        return {'leaf': True}
    if data.columns.size == 1:
        n = 0
        key = 'key'
        for name, g in group:
            if g.shape[0] > n:
                key = name
        node['type'] = key
        return {'leaf': True}
    node['leaf'] = False
    gr = 0
    att = 'attr'
    for c in data.columns.drop('haogua'):
        g = information_gain(data, c)
        print([c, g])
        if g > gr:
            gr = g
            att = c
    # print("--------------------", att, "---------------------")
    # print("--------------------", node['dat'], "---------------------")
    node['attr'] = att
    index = 1
    for names, g in data.groupby(att):
        node['branch'][str(index)] = generate_tree(g.drop(att, axis=1))
        index = index + 1
    return node


tree = generate_tree(ml)

nn = 0


def disp_tree(t, i):
    if t['leaf']:
        for k in t.keys():
            for j in range(0, i):
                print("----")
            print(k, ":", t[k])
    else:
        k = t['branch'].keys()
        i = i + 1
        for ki in k:
            disp_tree(t['branch'][ki], i)
    i = i - 1


# disp_tree(tree, nn)
print(tree)