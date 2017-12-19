# Classifier -- for each movie character network, these functions produce 
# 100 random graphs for each generative graph model and featurizes them. Then,
# classify_graph trains a model with a user-chosen classification algorithm,
# and it uses that model assign a random graph label to the original
# character network.

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

import graph_generators as gg
import feature_extractor as fe

# Choices for classification algorithm
classifier_choices = { 'SVC': SVC(), 
    'AdaBoost': AdaBoostClassifier(), 
    'KNeighbors': KNeighborsClassifier(), 
    'SGD': SGDClassifier() }

# Trains the classification model with training data x and y,
# using 20% of the training data as test data (note: as of present, 
# all algorithms use default hyperparameters)
# Here, x.shape = (num_samples, num_features) and 
# y.shape = (num_samples,)
def train_classifier(x, y, algo='SVC', train_prop=0.8):
    classifier = classifier_choices[algo]
    num_train = int(train_prop * x.shape[0])
    train_x = x[: num_train, :]
    train_y = y[: num_train]
    classifier.fit(train_x, train_y)
    train_accuracy = classifier.score(train_x, train_y)
    test_x = x[num_train: , :]
    test_y = y[num_train: ]
    test_accuracy = classifier.score(test_x, test_y)
    return classifier, train_accuracy, test_accuracy

# Produces labels for the test data given the classification model
# Here, classifier is some Classifier object from sklearn (with a
# predict method) and test_x is a numpy array with shape 
# (num_samples, num_features)
def test_classifier(classifier, test_x):
    return classifier.predict(test_x)

# Generates training graphs (based on the original graph, graph type 
# [DirectedGraphModel or UndirectedMultGraphModel], and feature names 
# [see the feature_extractor module]) for the classifier model to train on.
# Also calls test_classifier on the original graph
def classify_graph(orig_graph, graph_class, feature_names, 
    algo='KNeighbors', samples_per_category=100, draw_graphs=False, 
    verbose=False):
    categories = []
    categories = graph_class.__subclasses__()
    num_categories = len(categories)
    num_features = fe.get_features(orig_graph, feature_names).shape[0]

    data_x = np.zeros((num_categories * samples_per_category, num_features))
    data_y = np.zeros((num_categories * samples_per_category,))

    # Produces 100 graphs for each random graph process in [categories], 
    # making training data for the classifier
    orig_graph_obj = graph_class(orig_graph)
    if draw_graphs:
        orig_graph_obj.draw_graph(orig_graph_obj.nx_pos)
    for c in range(num_categories):
        for i in range(samples_per_category):
            new_directed_graph = categories[c](orig_graph)
            if draw_graphs:
                new_directed_graph.draw_graph(orig_graph_obj.nx_pos)
            if verbose and i == 0:
                new_directed_graph.summarize_metrics()
            data_x[samples_per_category * c + i, :] = fe.get_features(\
                new_directed_graph.graph, feature_names)
            data_y[samples_per_category * c + i] = c

    # Shuffle the training data
    permutation = range(len(data_x))
    random.shuffle(permutation)
    data_x = np.array([data_x[idx] for idx in permutation])
    data_y = np.array([data_y[idx] for idx in permutation])

    # Train the model and output the accuracies
    classifier, train_accuracy, test_accuracy = train_classifier( \
        data_x, data_y, algo=algo)

    # Output a label for the original character network
    test_x = np.zeros((1, num_features))
    test_x[0, :] = fe.get_features(orig_graph, feature_names)
    numerical_results = test_classifier(classifier, test_x)
    predictions = [categories[int(numerical_results[i])](orig_graph).name \
        for i in range(len(numerical_results))]

    return predictions, train_accuracy, test_accuracy



