import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.resnet import ResNet
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import sys

# set a high recursion limit so Threano doesnot complain
sys.setrecursionlimit(5000)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
    help="path to output checkpoint diretory")
ap.add_argument("-m", "--model", type=str,
    help="path to specific model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
    help="epoch to restart training at")
args = vars(ap.parse_args())

# load the train and test data, convert the images from int to float
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert the labels from int to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1,
    height_shift_range=0.1, horizontal_flip=True,
    fill_mode="nearest")

if args["model"] is None:
    print("[Info] compiling model...")
    opt = SGD(lr=1e-1)
    model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
               (64, 64, 128, 256), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
         metrics=["accuracy"])
else:
    print("[Info] load {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[Info] old learn rate:{}".format(
           K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-2)
    print("[Info] new learn rate:{}".format(
           K.get_value(model.optimizer.lr)))

callbacks = [
  EpochCheckpoint(args["checkpoints"], every=5,
    startAt=args["start_epoch"]),
  TrainingMonitor("output/resnet56_cifar10.png",
     jsonPath="output/resnet56_cifar10.json",
     startAt=args["start_epoch"])]

print("[Info] training network...")
model.fit_generator(
   aug.flow(trainX, trainY, batch_size=128),
   validation_data=(testX, testY),
   steps_per_epoch=len(trainX) // 128, epochs=80,
   callbacks=callbacks, verbose=1)
