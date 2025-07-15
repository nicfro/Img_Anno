from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
from collections import Counter
from ColorDistance import nearestColor
import json
import os

def find_most_prominent_colors(image_path, n_clusters=10):
    """Find the most prominent colors in an image using K-means clustering."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert the image from the RGB color space to the Lab color space
    # K-means clustering is based on euclidean distance, and Lab color space
    # provides perceptual meaning for euclidean distance
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Reshape the image into a feature vector so that k-means can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
     
    # Apply k-means using the specified number of clusters and
    # create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # Count how many times a label occurs and sort them in descending order
    occurrences_of_labels = Counter(labels)
    most_prominent_labels = sorted(occurrences_of_labels.keys(), key=lambda x: occurrences_of_labels[x], reverse=True)

    # Match the labels to their respective Lab color
    most_prominent_color_list = []
    for i in most_prominent_labels:
        item_index = np.where(labels == i)[0][0]
        most_prominent_color_list.append(quant[item_index])

    return most_prominent_color_list


def map_color_to_string_name(color_list):
    """Map Lab colors to their string names using the color dictionary."""
    if not os.path.exists("Color_Names_Dictionary.txt"):
        raise FileNotFoundError("Color_Names_Dictionary.txt not found. Run Reader/ReadColorFile.py first.")
    
    with open("Color_Names_Dictionary.txt", "r") as f:
        color_lookup = json.load(f)
    
    color_names = []
    for color in color_list:
        nearest_color_value = nearestColor(color)
        color_names.append(color_lookup[np.array_str(nearest_color_value)])
    
    return color_names


def main():
    """Main function to demonstrate the color extraction functionality."""
    image_path = "roundflower.jpg"
    
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Please provide a valid image path.")
        return
    
    try:
        prominent_colors = find_most_prominent_colors(image_path)
        color_names = map_color_to_string_name(prominent_colors)
        
        print(f"Most prominent colors in {image_path}:")
        for i, color_name in enumerate(color_names, 1):
            print(f"{i}. {color_name}")
            
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
