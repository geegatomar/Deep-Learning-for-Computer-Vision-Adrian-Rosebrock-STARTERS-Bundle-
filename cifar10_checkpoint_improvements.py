from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from minivggnet import MiniVGGNet
from keras.datasets import cifar10
import argparse
import numpy as np
import matplotlib.pyplot as plt

print("[INFO] loading data...")
((X_train, y_train), (X_test, y_test)) = cifar10.load_data()
X_train = X_train.astype("float")/255.0
X_test = X_test.astype("float")/255.0

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

fname = "cifar10_model_checkpoint_best_model.hdf5"
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks=[checkpoint]

print("[INFO] compiling model...")
model = MiniVGGNet.build(height = 32, width = 32, depth = 3, classes=10)
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=64, callbacks=callbacks)

print("[INFO] evaluating model...")
preds = model.predict(X_train, batch_size=64)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1)))




