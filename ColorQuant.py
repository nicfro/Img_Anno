"""
Color quantization and visualization script.
This script demonstrates image color quantization using K-means clustering
and displays the results with color swatches.
"""

from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
from collections import Counter
from ColorDistance import nearestColor
import json
import os
import argparse


def quantize_image(image_path, n_clusters=10):
    """Quantize an image using K-means clustering."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load the image and get its dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    
    # Convert the image from BGR to Lab color space
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

    # Count occurrences of each label and sort by frequency
    occurrences_of_labels = Counter(labels)
    most_prominent_labels = sorted(occurrences_of_labels.keys(), key=lambda x: occurrences_of_labels[x], reverse=True)

    # Get the most prominent colors
    most_prominent_color_list = []
    for i in most_prominent_labels:
        item_index = np.where(labels == i)[0][0]
        most_prominent_color_list.append(quant[item_index])
    
    return image, quant, most_prominent_color_list, h, w


def display_color_names(color_list):
    """Display the color names for a list of Lab colors."""
    if not os.path.exists("Color_Names_Dictionary.txt"):
        print("Warning: Color_Names_Dictionary.txt not found. Run Reader/ReadColorFile.py first.")
        return
    
    with open("Color_Names_Dictionary.txt", "r") as f:
        color_lookup = json.load(f)
    
    print("Most prominent colors:")
    for i, color in enumerate(color_list, 1):
        nearest_color_value = nearestColor(color)
        color_name = color_lookup[np.array_str(nearest_color_value)]
        print(f"{i}. {color_name}")


def create_color_palette(most_prominent_colors, height, width):
    """Create a color palette image from the most prominent colors."""
    interpolation = np.linspace(0, 1, len(most_prominent_colors) + 1).astype(float)
    blank_image = np.zeros((height, width, 3), np.uint8)

    # Create color swatches
    for i in range(len(interpolation) - 1):
        start_col = int(interpolation[i] * width)
        end_col = int(interpolation[i + 1] * width)
        blank_image[:, start_col:end_col] = most_prominent_colors[i]

    return blank_image


def main():
    """Main function to demonstrate color quantization."""
    parser = argparse.ArgumentParser(description="Color quantization and visualization")
    parser.add_argument("--image", default="roundflower.jpg", 
                       help="Path to the input image")
    parser.add_argument("--clusters", type=int, default=10, 
                       help="Number of color clusters")
    
    args = parser.parse_args()
    
    try:
        # Quantize the image
        image, quant, most_prominent_colors, h, w = quantize_image(args.image, args.clusters)
        
        # Display color names
        display_color_names(most_prominent_colors)
        
        # Create color palette
        color_palette = create_color_palette(most_prominent_colors, h, w)
        
        # Reshape the feature vectors back to images
        quant = quant.reshape((h, w, 3))
        image = image.reshape((h, w, 3))
        
        # Convert from Lab to BGR for display
        color_palette_bgr = cv2.cvtColor(color_palette, cv2.COLOR_LAB2BGR)
        quant_bgr = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        
        # Display the images side by side
        combined_image = np.hstack([image_bgr, quant_bgr, color_palette_bgr])
        cv2.imshow("Original | Quantized | Color Palette", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

