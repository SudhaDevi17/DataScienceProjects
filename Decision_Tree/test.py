from q1 import KNNClassifier as knc
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

knn_classifier = knc()
knn_classifier.train('./Datasets/q1/train.csv')
predictions = knn_classifier.predict('./Datasets/q1/test.csv')
test_labels = list()
with open("./Datasets/q1/test_labels.csv") as f:
    for line in f:
        test_labels.append(int(line))
print (accuracy_score(test_labels, predictions))

from q2 import KNNClassifier as knc

knn_classifier = knc()
knn_classifier.train('./Datasets/q2/train.csv')
predictions = knn_classifier.predict('./Datasets/q2/test.csv')
test_labels = list()
with open("./Datasets/q2/test_labels.csv") as f:
    for line in f:
        test_labels.append(line.strip())
print (accuracy_score(test_labels, predictions))

from q3 import DecisionTree as dtree

dtree_regressor = dtree()
dtree_regressor.train('./Datasets/q2/train.csv')
predictions = dtree_regressor.predict('./Datasets/q2/test.csv')
test_labels = list()
with open("./Datasets/q3/train.csv") as f:
    for line in f:
        test_labels.append(float(line.split(',')[1]))
print (mean_squared_error(test_labels, predictions))
