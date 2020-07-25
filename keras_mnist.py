from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from sklearn import datasets
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output plot")
args = vars(ap.parse_args())

print("[INFO] loading model...")
mnist = datasets.fetch_openml("mnist_784")
data = mnist.data.astype("float")/255.0
labels = mnist.target

le = LabelBinarizer()
labels = le.fit_transform(labels)


(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2)

print("[INFO] making model...")

model = Sequential()
model.add(Dense(256, input_shape=(784, ), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training model...")
#decay = 0.05/100
opt = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test) , epochs=100, batch_size=128)

print("[INFO] evaluating modek...")
preds = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=le.classes_))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig(args["output"])


