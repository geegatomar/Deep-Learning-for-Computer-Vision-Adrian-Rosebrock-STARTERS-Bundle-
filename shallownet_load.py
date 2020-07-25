from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from preprocessor import SimplePreprocessor
from datasetloader import SimpleDatasetLoader
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to i/p dataset")
ap.add_argument("-m", "--model", required=True, help="path to model")
args = vars(ap.parse_args())

imagePaths = np.array(list(paths.list_images(args["dataset"])))
ids = np.random.randint(0, len(imagePaths), size=(10, ))
imagePaths = imagePaths[ids]

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)

data = data.astype("float")/255.0

model = load_model(args["model"])

labelNames = ["cat", "dog", "panda"]

preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image,"Label: {}".format(labelNames[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)




