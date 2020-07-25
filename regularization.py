from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from preprocessor import SimplePreprocessor
from datasetloader import SimpleDatasetLoader
from imutils import paths
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.reshape((data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))
data = data.astype("float")/255.0

le = LabelEncoder()
labels = le.fit_transform(labels)

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2)

for r in (None, "l1", "l2"):
    print("[INFO] training model with {} regularization".format(r))
    model = SGDClassifier(loss="log", penalty = r, max_iter=10, learning_rate="constant", eta0=0.01)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("[INFO] accuracy in {} is {}".format(r, acc*100))






