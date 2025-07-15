import cv2
import numpy as np
import json
import pickle

def process_color_file():
    """Process color file and create Lab color dictionaries."""
    with open("Color_Names_From_Web.txt", "r") as colorFile:
        blank_image = np.zeros((1, 1, 3), np.uint8)
        resultDict = {}
        labColors = []
        currentColor = ""
        
        for line in colorFile:
            try:
                test = line.strip().replace("\t", ",").split(",")[0:2]
                test[1] = list(map(int, test[1].split(";")))
                test[1] = [test[1][2], test[1][1], test[1][0]]

                blank_image[0][0] = test[1]
                test[1] = cv2.cvtColor(blank_image, cv2.COLOR_BGR2LAB)[0][0]
                
                key = np.array_str(test[1])
                labColors.append(test[1])
                resultDict[key] = [currentColor, test[0]]

            except:
                currentColor = line.strip()
                continue
        
        with open("Color_Names_Dictionary.txt", "w") as f:
            json.dump(resultDict, f)
        
        with open("List_Of_Lab_Colors.txt", "wb") as f:
            pickle.dump(labColors, f)

if __name__ == "__main__":
    process_color_file()

