'''
python36 shallownet_load.py --dataset datasets/animals --model shallownet_weights.hdf5
'''

from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from keras.models import load_model
from imutils import paths
import numpy as np 
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",required=True,help="path to output model")
args = vars(ap.parse_args())

classLabels = ["cat","dog","panda"]

print("[INFO] sampling images...")
imagePaths = list(paths.list_images(args["dataset"]))
idxs = np.random.randint(0,len(imagePaths),size=(10,))

#print('imagePaths:', imagePaths)
print('idxs:', idxs)
#imagePaths = imagePaths[idxs]
imagePaths = np.array(imagePaths)[idxs]
#print('imagePaths:', imagePaths)

sp = simplepreprocessor.SimplePreprocessor(32,32)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp,iap])

(data,label) = sdl.load(imagePaths)

data = data.astype("float")/255.0

print("[INFO] loading pre-training network...")
model = load_model(args["model"])

print(dir(model))
print("[INFO] predicting...")
preds = model.predict(data,batch_size=32).argmax(axis=1)
#preds = model.predicting(data,batch_size=32).argmax(axis=1)

for(i,imagePath) in enumerate(imagePaths):

    print('imagePath:', imagePath)
    image = cv2.imread(imagePath)

    cv2.putText(image,"label:{}".format(classLabels[preds[i]]),(10,30),cv2.FONT_HERSHEY_SIMPLEX,
        0.7,(0,255,0),2)
    cv2.imshow("image",image)
    cv2.waitKey(0)
