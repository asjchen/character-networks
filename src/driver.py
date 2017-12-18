# Driver

import argparse
import random
import os
import csv

from data_processors import get_movie_networks
from classifier import classify_graph
import feature_extractor as fe
import graph_generators as gg

def main():
    parser = argparse.ArgumentParser(
        description=('Reads and converts movie dialog data to character '
            'networks, which are then classified as one of several random '
            'graph models (see README for more details)'))
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
    parser.add_argument('--draw_examples', '-d', action='store_true',
        help=('Draws examples of each random graph model, derived from '
        'the same character network'))
    parser.add_argument('--verbose', '-v', action='store_true', 
        help=('Print iteration numbers and summary metrics for each '
            'character network and its generated random graphs'))
    parser.add_argument('--output_predictions', '-o', 
        help=('Filename of CSV to store the labels of the movie '
            'character networks'))
    parser.add_argument('--feature', '-f', action='append', 
        choices=fe.feature_choices.keys(),
        help=('Adds features to consider for each graph among {}, default '
            'is {}'.format(fe.feature_choices.keys(), 
                fe.default_feature_names)))
    args = parser.parse_args()

    graph_class = gg.GraphModel
    if args.graph_type == 'multigraph':
        graph_class = gg.UndirectedMultiGraphModel
    elif args.graph_type == 'directed':
        graph_class = gg.DirectedGraphModel

    if args.feature is None:
        args.feature = fe.default_feature_names

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
    all_predictions = {}
    for i in range(len(movie_sample)):
        if args.verbose:
            print 'Iteration {}'.format(i)
            graph_class(movie_sample[i]).summarize_metrics()
        draw_graphs = False
        if args.draw_examples:
            draw_graphs = (i == 0) # draw graphs from first character network

        predictions, train_accuracy, test_accuracy = classify_graph( \
          movie_sample[i], graph_class, args.feature, 
          algo=args.classifier, draw_graphs=draw_graphs, verbose=args.verbose)

        if args.verbose:
            print 'Test Accuracy: {}'.format(test_accuracy)

        mean_train_accuracy += train_accuracy / len(movie_sample)
        mean_test_accuracy += test_accuracy / len(movie_sample)

        movie_name = movies[sample_indices[i]].name
        all_predictions[movie_name] = predictions[0]

    # Printing movie labels to output CSV if applicable
    if args.output_predictions is not None:
        with open(args.output_predictions, 'wb') as f:
            writer = csv.writer(f)
            movie_names = sorted(all_predictions.keys())
            movie_labels = [all_predictions[name] for name in movie_names]
            writer.writerow(['Movie Name', 'Label of Character Network'])
            writer.writerows(zip(movie_names, movie_labels))

    print 'Mean Train Accuracy: {}'.format(mean_train_accuracy)
    print 'Mean Test Accuracy: {}'.format(mean_test_accuracy)

if __name__ == '__main__':
    main()
