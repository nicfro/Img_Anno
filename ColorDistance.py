import numpy as np
from scipy.spatial import distance
import pickle

def nearestColor(some_pt):
    #Loading list of lab colors and converting to np array
    labColors = pickle.load(open("List_Of_Lab_Colors.txt","r"))
    labColors = np.array(labColors)

    #Finding and returning the closest pair in the lab color list to some point 
    return labColors[distance.cdist([some_pt], labColors).argmin()]

