# USAGE
# python predict_eyes.py --shape-predictor eye_predictor.dat

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import request
import urllib
import numpy as np

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into opencv format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="the image name")
args = vars(ap.parse_args())

image = args["image"]

# open the image
im = cv2.imread(image)
if im is None:
    im = url_to_image(image)

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

frame = imutils.resize(im, width=400)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale frame
rects = detector(gray, 0)
print('rects:', rects)

# loop over the face detections
for rect in rects:
        # convert the dlib rectangle into an OpenCV bounding box and
        # draw a bounding box surrounding the face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # use our custom dlib shape predictor to predict the location
        # of our landmark coordinates, then convert the prediction to
        # an easily parsable NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates from our dlib shape
        # predictor model draw them on the image
        for (sX, sY) in shape:
                cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)

# show the frame
cv2.imwrite("test.jpg", frame)

# do a bit of cleanup
cv2.destroyAllWindows()
