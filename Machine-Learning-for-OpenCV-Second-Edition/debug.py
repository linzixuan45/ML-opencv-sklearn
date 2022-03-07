import cv2
import numpy as np
import sklearn
from sklearn import datasets
import sklearn.model_selection
from sklearn import tree
import matplotlib.pyplot as plt

digits = datasets.load_digits()
dtree = tree.DecisionTreeClassifier()
dir(digits)
data ,feature_names,target = digits.data,digits.feature_names,digits.target

xTrain,xTest,yTrain,yTest = sklearn.model_selection.train_test_split(data,target,test_size=0.2,random_state=7)

max_depths = np.arange(1,100,1)
best_depth_score = []
for index,data in enumerate(max_depths):
    dtree = tree.DecisionTreeClassifier(max_depth=data)
    dtree.fit(xTrain,yTrain)
    score = dtree.score(xTest,yTest)
    best_depth_score.append(score)


min_samples_leaf  = np.arange(1,100,1)
best_min_samples_score = []
for index,data in enumerate(min_samples_leaf):
    dtree = tree.DecisionTreeClassifier(min_samples_leaf=data)
    dtree.fit(xTrain,yTrain)
    score = dtree.score(xTest,yTest)
    best_min_samples_score.append(score)


plt.plot(max_depths,best_min_samples_score,'o')
plt.show()
plt.plot(max_depths,best_depth_score,'o')
plt.show()

# best_dtree = tree.DecisionTreeClassifier(min_samples_leaf = min_samples_leaf[0],max_depth= max_depths[0])
# best_dtree.fit(xTrain,yTrain)
# print(best_dtree.score(xTest,yTest))