# Driver

import argparse

from data_processors import get_movie_networks
from classifier import classify_graph
from feature_extractor import get_eigenvalue_distribution, get_k_profiles
from graph_generators import DirectedGraphModel

def main():
    parser = argparse.ArgumentParser(
        description='Reads and processes the movie dialog data in a given directory')
    parser.add_argument('data_dir', help='Directory containing the dialog data')
    parser.add_argument('--classifier', '-c', help='Classifier algorithm: choose between SVC or Adaboost')
    args = parser.parse_args()

    movie_networks = get_movie_networks(args.data_dir)

    small_sample = movie_networks.values()[:20]
    mean_train_accuracy = 0.0
    mean_test_accuracy = 0.0
    all_predictions = []
    for i in range(len(small_sample)):
        print i
        DirectedGraphModel(small_sample[i]).summarize_metrics()
        # predictions, train_accuracy, test_accuracy = classify_graph( \
        #     small_sample[i], get_eigenvalue_distribution, algo=args.classifier)
        predictions, train_accuracy, test_accuracy = classify_graph( \
            small_sample[i], get_k_profiles, algo=args.classifier)
        mean_train_accuracy += train_accuracy / len(small_sample)
        mean_test_accuracy += test_accuracy / len(small_sample)
        all_predictions.append(predictions[0])
    print all_predictions
    print 'Mean Train Accuracy: {}'.format(mean_train_accuracy)
    print 'Mean Test Accuracy: {}'.format(mean_test_accuracy)

if __name__ == '__main__':
    main()
