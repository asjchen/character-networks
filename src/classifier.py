# Classifier

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


def train_classifier(x, y, algo='SVC', train_prop=0.8):
    if algo == 'SVC':
        classifier = SVC()
    elif algo == 'AdaBoost':
        classifier = AdaBoostClassifier()
    elif algo == 'KNeighbors':
        classifier = KNeighborsClassifier()
    elif algo == 'SGD':
        classifier = SGDClassifier()
    num_train = int(train_prop * x.shape[0])
    train_x = x[: num_train, :]
    train_y = y[: num_train]
    classifier.fit(train_x, train_y)
    train_accuracy = classifier.score(train_x, train_y)
    test_x = x[num_train: , :]
    test_y = y[num_train: ]
    test_accuracy = classifier.score(test_x, test_y)
    # print confusion_matrix(test_y, classifier.predict(test_x))
    return classifier, train_accuracy, test_accuracy

def test_classifier(classifier, test_x):
    return classifier.predict(test_x)

def classify_graph(orig_graph, graph_class, feature_extractor, algo='SVC', samples_per_category=100, draw_graphs=False):
    categories = []
    if graph_class == gg.DirectedGraphModel:
        categories = [gg.DirectedErdosRenyi, gg.DirectedChungLu, \
            gg.FastReciprocalDirected, gg.DirectedPreferentialAttachment]
    elif graph_class == gg.UndirectedMultiGraphModel:
        categories = [gg.MultiPreferentialAttachment, gg.MultiConfiguration, \
            gg.MultiErdosRenyi, gg.MultiChungLu]
    num_categories = len(categories)
    num_features = feature_extractor(orig_graph).shape[0]

    data_x = np.zeros((num_categories * samples_per_category, num_features))
    data_y = np.zeros((num_categories * samples_per_category,))

    orig_graph_obj = graph_class(orig_graph)
    if draw_graphs:
        orig_graph_obj.draw_graph(orig_graph_obj.nx_pos)
    for c in range(num_categories):
        for i in range(samples_per_category):
            new_directed_graph = categories[c](orig_graph)
            if draw_graphs:
                new_directed_graph.draw_graph(orig_graph_obj.nx_pos)
                if i == 0:
                    new_directed_graph.summarize_metrics()
            data_x[samples_per_category * c + i, :] = feature_extractor(new_directed_graph.graph)
            data_y[samples_per_category * c + i] = c

    permutation = range(len(data_x))
    random.shuffle(permutation)
    data_x = np.array([data_x[idx] for idx in permutation])
    data_y = np.array([data_y[idx] for idx in permutation])

    classifier, train_accuracy, test_accuracy = train_classifier(data_x, data_y, algo=algo)

    test_x = np.zeros((1, num_features))
    test_x[0, :] = feature_extractor(orig_graph)
    numerical_results = test_classifier(classifier, test_x)
    predictions = [categories[int(numerical_results[i])](orig_graph).name for i in range(len(numerical_results))]

    return predictions, train_accuracy, test_accuracy



