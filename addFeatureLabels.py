import numpy as np
import pandas as pd
import random
import sys, os
import csv

features_cols = [None]*14   # Column => feature name
DATA_DIR = "./processed_data"
VISTA_labels = "./Enhancer_Prediction/Tables/enhancers.xlsx"

def add_vista_labels(features):
    """Relabel each row in the numpy array with the VISTA identifier based on limb region or not."""
    df = pd.read_excel(VISTA_labels)
    activity = df["Limb-activity"]
    f = lambda label: -1 if label == "negative" else 1
    labels = np.array([[f(label) for label in reversed(activity)]])
    labels = labels.transpose()

    labeled_features = np.concatenate((features, labels), axis=1)
    np.save("feature_matrix_w_labels", labeled_features)
    return labeled_features


def get_labels():
    df = pd.read_excel(VISTA_labels)
    activity = df["Limb-activity"]
    f = lambda label: -1 if label == "negative" else 1
    labels = np.array([f(label) for label in reversed(activity)])
    return labels.transpose()


def get_vista_labels():
    df = pd.read_csv("enhancers.csv")
    labels = df.values

    return labels


def read_files():
    """Create numpy array from the csv files"""
    files = os.listdir(DATA_DIR)
    features = [[None for _ in range(2203)] for __ in range(13)]

    for i, filename in enumerate(files):
        csv_str = DATA_DIR + '/' + filename

        print("-------------------------------------------")
        print("                                           ")
        print("Processing file ", i, " : ", filename)

        with open(csv_str, 'rU') as csvfile:
            feature = csv.reader(csvfile, delimiter=",")
            features[i] = feature.next()

        features_cols[12 - i] = filename

    features = np.array(features)
    features = features.transpose()

    features_cols[13] = "label"
    labeled_features = add_vista_labels(features)
    return features


# features = read_files()
# labels = get_labels()
