from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from lenet import LeNet
from keras.optimizers import SGD
from sklearn import datasets
from keras import backend as K
import numpy as np
import argparse 
import cv2



dataset = datasets.fetch_openml("mnist_784")
data = dataset.data.astype("float")/255.0
labels = dataset.target

if K.image_data_format() == "channels_first":
    data = data.reshape((data.shape[0], 1, 28, 28))
else:
    data = data.reshape((data.shape[0], 28, 28, 1))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25)

print("[INFO] building model...")
model = LeNet.build(width=28, height=28, depth=1, classes=10)

print("[INFO] training network...")
opt = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

print("[INFO] evaluating model...")
preds = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

