# USAGE
# python classify_iris.py
# python classify_iris.py --model svm

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="knn",
        help="type of python machine learning model to use")
args = vars(ap.parse_args())

# define the dictionary of models our script can use, where the key
# to the dictionary is the name of the model (supplied via command
# line argument) and the value is the model itself
models = {
        "knn": KNeighborsClassifier(n_neighbors=1),
        "naive_bayes": GaussianNB(),
        "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
        "svm": SVC(kernel="rbf", gamma="auto"),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "mlp": MLPClassifier()
}

# load the Iris dataset and perform a training and testing split,
# using 75% of the data for training and 25% for evaluation
print("[INFO] loading data...")
dataset = load_iris()
(trainX, testX, trainY, testY) = train_test_split(dataset.data,
        dataset.target, random_state=3, test_size=0.25)

# train the model
print("[INFO] using '{}' model".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)

# make predictions on our data and show a classification report
print("[INFO] evaluating...")
predictions = model.predict(testX)
print(classification_report(testY, predictions,
        target_names=dataset.target_names))
        
"""
root@a700e846c4ce:/nn/startCv/python-machine-learning# python classify_iris.py --model naive_bayes
[INFO] loading data...
[INFO] using 'naive_bayes' model
[INFO] evaluating...
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        15
  versicolor       1.00      0.92      0.96        12
   virginica       0.92      1.00      0.96        11

    accuracy                           0.97        38
   macro avg       0.97      0.97      0.97        38
weighted avg       0.98      0.97      0.97        38
"""
