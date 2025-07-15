import numpy as np
from scipy.spatial import distance
import pickle

def nearestColor(some_pt):
    """Find the nearest color in the Lab color space to a given point."""
    # Loading list of lab colors and converting to np array
    with open("List_Of_Lab_Colors.txt", "rb") as f:
        labColors = pickle.load(f)
    labColors = np.array(labColors)

    # Finding and returning the closest pair in the lab color list to some point 
    return labColors[distance.cdist([some_pt], labColors).argmin()]

