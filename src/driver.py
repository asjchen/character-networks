# Driver

import argparse
import random
import os

from data_processors import get_movie_networks
from classifier import classify_graph
import feature_extractor as fe
import graph_generators as gg

def main():
    parser = argparse.ArgumentParser(
        description=('Reads and converts movie dialog data to character '
            'networks, which are then classified as one of several random '
            'graph models'))
    parser.add_argument('data_dir', help='Directory containing dialog data')
    parser.add_argument('--classifier', '-c', default='SVC', 
        help='Classifier algorithm: choose between SVC or Adaboost')
    parser.add_argument('--graph_type', '-g', default='multigraph', 
        choices=['multigraph', 'directed'], 
        help=('Graph type to be created, choices are multigraph (each '
            'undirected edge represents one conversation) and directed '
            '(edge from A to B if B talks more than A in at least one '
            'conversation, resulting in a simple directed graph)'))
    parser.add_argument('--sample', '-s', type=int, 
        help=('Randomly choose a sample of the specified number of '
            'movies to evaluate'))
    # TODO: option to draw the first graph
    # TODO: option to print results to csv
    # TODO: add requirements.txt
    # TODO: add verbose flag
    # TODO: add option to select multiple features and have functions to combine them
    args = parser.parse_args()

    graph_class = gg.GraphModel
    if args.graph_type == 'multigraph':
        graph_class = gg.UndirectedMultiGraphModel
    elif args.graph_type == 'directed':
        graph_class = gg.DirectedGraphModel

    # Make bin directory if it doesn't exist
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    bin_dir = os.path.join(project_dir, 'bin')
    if not os.path.isdir(bin_dir):
        os.mkdir(bin_dir)

    movies, movie_networks = get_movie_networks(args.data_dir, graph_class)
    randomized_keys = movie_networks.keys()
    random.shuffle(randomized_keys)
    sample_indices = randomized_keys[:]
    if args.sample is not None:
        sample_indices = randomized_keys[: args.sample]

    movie_sample = [movie_networks[i] for i in sample_indices]
    mean_train_accuracy = 0.0
    mean_test_accuracy = 0.0
    all_predictions = []
    for i in range(len(movie_sample)):
        # TODO: only print when verbose
        print i
        if i == 0:
            graph_class(movie_sample[i]).summarize_metrics()
        #draw_graphs = (i == 0)
        draw_graphs=False

        predictions, train_accuracy, test_accuracy = classify_graph( \
          movie_sample[i], graph_class, fe.get_multigraph_features, 
          algo=args.classifier, draw_graphs=draw_graphs)

        # TODO: only if verbose
        # print '{}'.format(test_accuracy)

        mean_train_accuracy += train_accuracy / len(movie_sample)
        mean_test_accuracy += test_accuracy / len(movie_sample)
        all_predictions.append(predictions[0])
    # TODO: change this to summarize the predictions and print results in CSV if output flag given
    print all_predictions
    print 'Mean Train Accuracy: {}'.format(mean_train_accuracy)
    print 'Mean Test Accuracy: {}'.format(mean_test_accuracy)

if __name__ == '__main__':
    main()
