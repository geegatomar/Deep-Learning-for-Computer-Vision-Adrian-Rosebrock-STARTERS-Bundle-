from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from shallownet import ShallowNet
from keras.datasets import cifar10
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

((X_train, y_train), (X_test, y_test)) = cifar10.load_data()

X_train = X_train.astype("float")/255.0
X_test = X_test.astype("float")/255.0

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

print("[INFO] training network...")
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
opt = SGD(0.005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=32)

print("[INFO] evaluating network...")
preds = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1)))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
