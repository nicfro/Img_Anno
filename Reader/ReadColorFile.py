import cv2
import numpy as np
import json
import pickle

colorFile = open("Color_Names_From_Web.txt", "r")
blank_image = np.zeros((1,1,3), np.uint8)
resultDict = {}
labColors = []
for line in colorFile:
    try:
        test = str.strip(line).replace("\t",",").split(",")[0:2]
        #print test
        test[1] = map(int,test[1].split(";"))
        test[1] = [test[1][2],test[1][1],test[1][0]]

        blank_image[0][0] = test[1]

            
        test[1] = cv2.cvtColor(blank_image, cv2.COLOR_BGR2LAB)[0][0]
        
        key = np.array_str(test[1])
        labColors.append(test[1])
        resultDict[key] = [currentColor,test[0]] 


    except:
        currentColor = line.strip()
        continue
    
json.dump(resultDict, open("Color_Names_Dictionary.txt","w"))
pickle.dump(labColors, open("List_Of_Lab_Colors.txt","w"))

