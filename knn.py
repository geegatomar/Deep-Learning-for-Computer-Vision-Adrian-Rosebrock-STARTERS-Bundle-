from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessor import SimplePreprocessor
from datasetloader import SimpleDatasetLoader
from imutils import paths
import argparse
import numpy as np
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=3, help="no of neighbors")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="no of jobs for knn distance")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
data, labels = sdl.load(imagePaths, verbose=500)

data = data.reshape((data.shape[0], 32*32*3))

le = LabelEncoder()
labels = le.fit_transform(labels)

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25)

print("[INFO] training model...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))




