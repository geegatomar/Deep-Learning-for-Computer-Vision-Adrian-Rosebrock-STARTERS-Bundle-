import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
original = image

#image = cv2.resize(original, (32, 32)).flatten()
#image = cv2.resize(original, (32, 32)).resize((32*32*3, 1))
image = cv2.resize(original, (32, 32))
image = np.resize(image, (32*32*3, 1))

print(image.shape)
W = np.random.randn(3, 3072)
b = np.random.randn(3, 1)

labels = ["cat", "dog", "panda"]

score = np.dot(W, image) + b
#score = W.dot(image) + b

print(W.shape, image.shape, b.shape)
print(score)

cv2.putText(original, "Label : {}".format(labels[np.argmax(score)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("img", original)
cv2.waitKey(0)



