from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
from collections import Counter
from ColorDistance import nearestColor
import json

def FindMostProminentColors(image):
    image = cv2.imread(image)
    (h, w) = image.shape[:2]
     
    # convert the image from the RGB color space to the Lab
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # Lab color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
     
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = 10)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    #Counting how many times a label occurs and sorting them in descending order
    occurancesOfLabels = Counter(labels)
    mostProminentLabels = sorted(occurancesOfLabels, key=occurancesOfLabels.get, reverse=True)

    #Matching the labels to their respective Lab color
    mostProminentColorList = []
    for i in mostProminentLabels:
        itemIndex = np.where(labels==i)[0][0]
        mostProminentColorList.append(quant[itemIndex])

    return mostProminentColorList


def MapColorToStringName(ColorList):
    colorNames = []
    for i in range(len(ColorList)):
        nearestColorValue = nearestColor(ColorList[i])
        colorLookup = json.load(open("Color_Names_Dictionary.txt","r"))
        colorNames.append(colorLookup[np.array_str(nearestColorValue)])
    return colorNames

test = MapColorToStringName(FindMostProminentColors("roundflower.jpg"))
