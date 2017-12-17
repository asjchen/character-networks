# Driver

import argparse
import random

from data_processors import get_movie_networks
from classifier import classify_graph
import feature_extractor as fe
import graph_generators as gg

def main():
    parser = argparse.ArgumentParser(
        description='Reads and processes the movie dialog data in a given directory')
    parser.add_argument('data_dir', help='Directory containing the dialog data')
    parser.add_argument('--classifier', '-c', default='SVC', help='Classifier algorithm: choose between SVC or Adaboost')
    parser.add_argument('--graph_type', '-g', default='multigraph', choices=['multigraph', 'directed'], 
        help='Graph type to be created, choices are multigraph and directed')
    args = parser.parse_args()

    graph_class = gg.GraphModel
    if args.graph_type == 'multigraph':
        graph_class = gg.UndirectedMultiGraphModel
    elif args.graph_type == 'directed':
        graph_class = gg.DirectedGraphModel

    movies, movie_networks = get_movie_networks(args.data_dir, graph_class)
    randomized_keys = movie_networks.keys()
    random.shuffle(randomized_keys)
    small_indices = randomized_keys[:20]
    small_sample = [movie_networks[i] for i in small_indices]
    mean_train_accuracy = 0.0
    mean_test_accuracy = 0.0
    all_predictions = []
    for i in range(len(small_sample)):
        print i
        if i == 0:
            graph_class(small_sample[i]).summarize_metrics()
        #draw_graphs = (i == 0)
        draw_graphs=False

        predictions, train_accuracy, test_accuracy = classify_graph( \
          small_sample[i], graph_class, fe.get_multigraph_features, 
          algo=args.classifier, draw_graphs=draw_graphs)

        print '{}'.format(test_accuracy)

        mean_train_accuracy += train_accuracy / len(small_sample)
        mean_test_accuracy += test_accuracy / len(small_sample)
        all_predictions.append(predictions[0])
    print all_predictions
    print 'Mean Train Accuracy: {}'.format(mean_train_accuracy)
    print 'Mean Test Accuracy: {}'.format(mean_test_accuracy)

if __name__ == '__main__':
    main()
