import sklearn.datasets
from pathlib import Path
import subprocess
import os
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

'''加载数据集'''
iris = sklearn.datasets.load_iris()
feature_names = iris.feature_names
# 标签有三个['setosa', 'versicolor', 'virginica'] 山鸢尾、变色鸢尾和维吉尼亚鸢尾
'''初始化数据'''
Data, target = iris.data, iris.target
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(Data, target, test_size=0.1, random_state=7)
'''初始化树'''
dtree = tree.DecisionTreeClassifier()
dtree.fit(xTrain, yTrain)

'''显示生成的决策树'''

