# Classifier

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import random

import graph_generators as gg


def train_classifier(x, y, algo='AdaBoost', train_prop=0.8):
    if algo == 'SVC':
        classifier = SVC()
    elif algo == 'AdaBoost':
        classifier = AdaBoostClassifier()
    num_train = int(train_prop * x.shape[0])
    train_x = x[: num_train, :]
    train_y = y[: num_train]
    classifier.fit(train_x, train_y)
    train_accuracy = classifier.score(train_x, train_y)
    test_x = x[num_train: , :]
    test_y = y[num_train: ]
    test_accuracy = classifier.score(test_x, test_y)
    return classifier, train_accuracy, test_accuracy

def test_classifier(classifier, test_x):
    return classifier.predict(test_x)

def classify_graph(orig_graph, feature_extractor, samples_per_category=200):
    # TODO: make categories global
    # categories = [gg.DirectedErdosRenyi, gg.DirectedChungLu, gg.DirectedConfiguration]
    categories = [gg.DirectedErdosRenyi, gg.DirectedChungLu, gg.FastDirectedReciprocal]
    num_categories = len(categories)
    num_features = feature_extractor(orig_graph).shape[0]

    data_x = np.zeros((num_categories * samples_per_category, num_features))
    data_y = np.zeros((num_categories * samples_per_category,))

    for c in range(num_categories):
        for i in range(samples_per_category):
            new_graph = categories[c](orig_graph).graph
            data_x[samples_per_category * c + i, :] = feature_extractor(new_graph)
            data_y[samples_per_category * c + i] = c

    permutation = range(len(data_x))
    random.shuffle(permutation)
    data_x = np.array([data_x[idx] for idx in permutation])
    data_y = np.array([data_y[idx] for idx in permutation])

    classifier, train_accuracy, test_accuracy = train_classifier(data_x, data_y)

    test_x = np.zeros((1, num_features))
    test_x[0, :] = feature_extractor(orig_graph)
    numerical_results = test_classifier(classifier, test_x)
    predictions = [categories[i](orig_graph).name for i in range(len(numerical_results))]
    return predictions, train_accuracy, test_accuracy



