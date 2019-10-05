import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

def extract_color_stats(image):
    # split the input image into its respective RGB color channels
    # and then create a feature vector with 6 values: the mean and 
    # standard deviation for each of the 3 channels, respectively
    (R, G, B) = cv2.split(image)
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]

    # return our set of features
    return features

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
  help="path to directory containing the 3scenes dataset")
ap.add_argument("-m", "--model", type=str, default="knn",
  help="type of python machine learning model to use")
args = vars(ap.parse_args())

models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}

imagePaths = paths.list_images(args["dataset"])
data = []
labels = []

for imagePath in imagePaths:
    #image = Image.open(imagePath)
    image = cv2.imread(imagePath)
    features = extract_color_stats(image)
    data.append(features)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
  test_size=0.25)

model = models[args["model"]]
model.fit(trainX, trainY)

predictions = model.predict(testX)
print(classification_report(testY, predictions,
    target_names=le.classes_))

# 输出
root@deepface-19:/data/zhouping/anaconda3/nn/dl4cv# python36 knn_colorStats.py --dataset datasets/animals
              precision    recall  f1-score   support

        cats       0.43      0.43      0.43       247
        dogs       0.41      0.41      0.41       235
       panda       0.67      0.66      0.66       268

    accuracy                           0.51       750
   macro avg       0.50      0.50      0.50       750
weighted avg       0.51      0.51      0.51       750

# 数据
链接：https://pan.baidu.com/s/1Ij2dvmM9aq-fNjrhq2PYgg 
提取码：jwco 
