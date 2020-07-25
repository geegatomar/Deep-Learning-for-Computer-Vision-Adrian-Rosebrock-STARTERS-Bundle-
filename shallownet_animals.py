from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessor import SimplePreprocessor
from datasetloader import SimpleDatasetLoader
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.astype("float")/255.0

labelNames = ["cat", "dog", "panda"]

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25)

y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)


print("[INFO] training network...")
model = ShallowNet.build(height=32, width=32, depth=3, classes=3)

opt = SGD(lr=0.003)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

print("[INFO] evaluating model...")
preds = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames))

import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

