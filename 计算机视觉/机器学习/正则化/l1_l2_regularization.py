# import the packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

# parse script arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)

# 3072 = 32 * 32 * 3
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of the
# data for training and the remaining 25% for testing.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

# try a few different regularization techniques
for r in (None, "l1", "l2"):
    # train a SGD classifier using softmax loss function and the
    # specified regularization function for 10 epochs
    print("[INFO] training model with '{}' penalty".format(r))

    model = SGDClassifier(loss="log", penalty=r, max_iter=10,
        learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)

    print("acc", acc)

    print("[INFO] '{}' penalty accuracy: {:.2f}%".format(r, acc*100))
    
#输出
root@deepface-19:/data/zhouping/anaconda3/nn/dl4cv# python36 regularization.py --dataset datasets/animals
/data/zhouping/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
[INFO] loading images...
[INFO] processed 500/3000
[INFO] processed 1000/3000
[INFO] processed 1500/3000
[INFO] processed 2000/3000
[INFO] processed 2500/3000
[INFO] processed 3000/3000
[INFO] training model with 'None' penalty
/data/zhouping/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
  ConvergenceWarning)
acc 0.5786666666666667
[INFO] 'None' penalty accuracy: 57.87%
[INFO] training model with 'l1' penalty
/data/zhouping/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
  ConvergenceWarning)
acc 0.5213333333333333
[INFO] 'l1' penalty accuracy: 52.13%
[INFO] training model with 'l2' penalty
/data/zhouping/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:561: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
  ConvergenceWarning)
acc 0.5146666666666667
[INFO] 'l2' penalty accuracy: 51.47%
