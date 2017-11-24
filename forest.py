@@ -1,208 +0,0 @@
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
import sys, os
import csv

from sklearn import svm, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve
from sklearn.cross_validation import cross_val_score

from addFeatureLabels import *
from pylab import *

def report_metrics(y_test, y_pred_lbl):
    '''
    Return Precision/Recall/accuracy
    INPUT: y_test: Array of true labels
       y_pred_lbl: Array of predicted labels
    OUTPUT: Return precision, recall and accuracy score values
    '''
    precision = precision_score(y_test, y_pred_lbl)
    recall = recall_score(y_test, y_pred_lbl)
    accuracy = accuracy_score(y_test, y_pred_lbl)
    return precision, recall, accuracy

def plot_roc(y_test, y_pred_prod, name):
    '''
    Using sklearn roc_curve plot roc curve
    INPUT:
    y_test: Array of true labels
    y_pred_prod: Array of probabilities of target variable
    OUTPUT:
    None
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prod)
    plt.plot(fpr, tpr, label = name)
    plt.rcParams['font.size'] = 12
    #plt.title('ROC curve for Churn Classifier')
    plt.xlabel('Rate of False Enhancer Identification')
    plt.ylabel('Rate of True Enhancer Identification')
    plt.grid(True)

def get_num_features(data):
    return len(data)

def create_splits(pos_examples, neg_examples, SPLIT=0.8):
    results = []
    indices = list(range(len(results)))
    random.shuffle(indices)

def separate_data(data, labels):
    pos_examples, neg_examples = [], []
    for i, label in enumerate(labels):
        if label == 1:
            pos_examples.append(data[i])
        else:
            neg_examples.append(data[i])

    return pos_examples, neg_examples

def split_unequal(data, labels, SPLIT=0.106):
    pos_examples, neg_examples = separate_data(data, labels)

    true_split = float(len(pos_examples)) / (len(neg_examples) + len(pos_examples))
    print "This is the actual splitting of the data: ", true_split

    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(pos_examples)):
        if i <= float(len(pos_examples))*0.8:

            num = random()
            if num >= SPLIT:
                X_train.append(pos_examples[i])
                y_train.append(1)

            else:
                X_train.append(neg_examples[i])
                y_train.append(-1)

        else:

            if random() >= SPLIT:
                X_test.append(pos_examples[i])
                y_test.append(1)

            else:
                X_test.append(neg_examples[i])
                y_test.append(-1)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def split_equal(data, labels):
    pos_examples, neg_examples = separate_data(data, labels)

    true_split = float(len(pos_examples)) / len(neg_examples)
    print "This is the actual splitting of the data: ", true_split

    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(pos_examples)):
        if i <= float(len(pos_examples))*0.8:
            X_train.append(pos_examples[i])
            y_train.append(1)

            X_train.append(neg_examples[i])
            y_train.append(-1)

        else:
            X_test.append(pos_examples[i])
            y_test.append(1)

            X_test.append(neg_examples[i])
            y_test.append(-1)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def train_step(data, labels, model, name):
    X_train, X_test, y_train, y_test = split_unequal(data, labels)

    model.fit(X_train, y_train)
    print model.get_params()

    train_predictions = model.predict(X_train)
    print "These is the training accuracy  ", report_metrics(y_train, train_predictions)

    predictions = model.predict(X_test)
    print "Here are the predictions for ", name, " : ", predictions

    print report_metrics(y_test, predictions)

    predict_label = model.predict_proba(X_test)[:,1]
    # predic_label = model.predict_log_proba(X_test)[:,1]
    plot_roc(y_test, predict_label, name)
    plt.legend()
    plt.title(name + "ROC curve " + "Split=0.106")
    plt.show()

def train(data, labels, model, name):
    X_train, X_test, y_train, y_test = train_test_split(data,labels)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print "Here are the predictions for ", name, " : ", predictions

    print report_metrics(y_test, predictions)

    predict_label = model.predict_proba(X_test)[:,1]
    plot_roc(y_test, predict_label, name)
    plt.legend()
    plt.title(name + " ROC curve " + "Split=0.40")
    plt.show()

data = read_files()
data = data.astype(float)
labels = get_labels()
labels = labels.astype(float)

train_step(data, labels, RandomForestClassifier(), "Random Forest")
train(data, labels, RandomForestClassifier(), "Random Forest")

multiple_features = np.array([])
for col in range(5):
    single_feature = data[:,col]
    print single_feature, single_feature.shape
    print labels, labels.shape

    lr = LogisticRegression()
    lr.fit(single_feature, labels)
    break

    train_step(single_feature, labels, RandomForestClassifier(), "Random Forest")
    multiple_features = np.append(multiple_features, single_feature)
    train_step(multiple_features, labels, RandomForestClassifier(), "Random Forest")

print multiple_features, multiple_features.shape
train_step(multiple_features, labels, RandomForestClassifier(), "Random Forest")

print "Examples ", data
print "Labels: ", labels

models = [RandomForestClassifier()]
for i, model in enumerate(models):
    name = "Random Forest"
    train(data, labels, model, name)

logistic_accuracy = cross_val_score(models[0], data, labels, cv=10, scoring="accuracy")
print logistic_accuracy

logistic_precision = cross_val_score(models[0], data, labels, cv=10, scoring="precision")
print logistic_precision

logistic_recall = cross_val_score(models[0], data, labels, cv=10, scoring="recall")
print logistic_recall

lasso = linear_model.Lasso()
y_pred = cross_val_predict(lasso, data, labels, cv=10)

fig, ax = plt.subplots()
ax.scatter(labels, y_pred, edgecolors=(0, 0, 0))
ax.plot([-2, 2], [-2, 2], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
