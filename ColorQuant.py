 # -*- coding: cp1252 -*-
# import the necessary packages
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
import operator
from collections import Counter
from ColorDistance import nearestColor
import json

#img = "824c8039a89feb1ad84f577c7f224c22.jpg"
#img = "11507566_8.jpg"
img = "roundflower.jpg"
#img = "27806320123_60e5fa694b_h.jpg"

# load the image and grab its width and height
image = cv2.imread(img)
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

occurancesOfLabels = Counter(labels)
mostProminentLabels = sorted(occurancesOfLabels, key=occurancesOfLabels.get, reverse=True)

mostProminentColorList = []
for i in mostProminentLabels:
    itemIndex = np.where(labels==i)[0][0]
    mostProminentColorList.append(quant[itemIndex])
    
for i in range(len(mostProminentColorList)): 
    nearestColorValue = nearestColor(mostProminentColorList[i])
    colorLookup = json.load(open("Color_Names_Dictionary.txt","r"))
    print colorLookup[np.array_str(nearestColorValue)]

# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))

#defining heigh and width of color image
height = np.shape(image)[0]
width = np.shape(image)[1]

interpolation = np.linspace(0, 1, len(mostProminentColorList)+1).astype(float).tolist()
blank_image = np.zeros((height,width,3), np.uint8)

'''
x = Starting point
y = End point
[:, x*width : y*width]
'''
for i in range(len(interpolation)-1):
    blank_image[:,interpolation[i]*width : interpolation[i+1]*width] = mostProminentColorList[i]

blank_image_upd = cv2.cvtColor(blank_image, cv2.COLOR_LAB2BGR)

# convert from Lab to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

# display the images and wait for a keypress
cv2.imshow("image", np.hstack([image, quant, blank_image_upd]))
cv2.waitKey(0)

