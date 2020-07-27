from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from minivggnet import MiniVGGNet
from keras.datasets import cifar10
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to output file")
args = vars(ap.parse_args())

print("[INFO] loading data...")
((X_train, y_train), (X_test, y_test)) = cifar10.load_data()

X_train = X_train.astype("float")/255.0
X_test = X_test.astype("float")/255.0

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

print("[INFO[ loading model...")
model = MiniVGGNet.build(height=32, width=32, depth=3, classes=10)
opt = SGD(0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=64)

model.save('minivggnet_cifar10_with_BN.hdf5')

print("[INFO] evaluating model...")
preds = model.predict(X_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1)))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])


