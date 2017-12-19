# Modeling Movie Character Networks with Random Graphs

This set of scripts attempts to characterize the character networks of 617 
movies, taken from the Cornell Movie Dialogs Corpus (accessible on Kaggle at
https://www.kaggle.com/Cornell-University/movie-dialog-corpus). 

More specifically, it attempts to find which generative processes for random 
graphs (such as Preferential Attachment or Erdos-Renyi) are most likely to 
produce the movie character networks. Such information provides high-level 
insight into the films' structures and might tell us how a writer might 
develop characters and their interactions with each other. 

In this scenario, we are specifically interested in directed and multigraph 
character networks, as such models provide more information than simple 
undirected networks would. (See Bonato et al. 2016 for a description of the
process for simple undirected character networks.)

At a high level, the classification process for each movie is as follows:
* Construct the character network: node for every character
    * If we're interested in multigraphs, then we create an edge for every
    conversation that two characters have (see `imgs/multigraph.png` for an 
    example)
    * If we're interested in directed (simple) graphs, then we create an 
    edge from character A to character B if B ever talks more than A in a
    conversation (see `imgs/directed_graph.png` for an example)
* Generate random graphs: create 100 samples of each graph model from the 
original character network
* Classify random graphs: featurize the random graphs and train a classifier
* Categorize the original character network

For a visual summary, see `imgs/master_plan.png`.


## Requirements
* Snap.py: install version 4.0 or later with the instructions here: 
http://snap.stanford.edu/snappy/ (Note: do not install via pip, as pip 
only allows installing earlier versions of Snap.py)
* Other Python Libraries: the remaining required libraries may be installed 
via Pip with the following command:

```
pip install -r requirements.txt
```


## Running Scripts

Migrate to the src/ directory and run the following command:
```
python driver.py ../data/movie-dialog-corpus/ 
```
The optional arguments are as follows:
* `-c` / `--classifier`: the classifier algorithm to be used (one of SVC, 
AdaBoost, KNeighbors, or SGD), default is KNeighbors
* `-d` / `--draw_examples`: produces example images of character networks and
generated random graphs in the bin directory
* `-f [FEATURE]` / `--feature [FEATURE]`: adds features to consider for each 
graph (so this flag can be used multiple times), choices are:
    * GraphProfiles: examines the topologies of graphlets / graph profiles
    * SpectralHistogram: looks at the histogram of spectral eigenvalues
    * BetweennessHistogram: uses betweenness centralities of nodes and edges
    * NodeCentrality: computes several features relating to node centrality
    * PageRank: finds the histogram of normalized PageRank values of the nodes
* `-g` / `--graph_type`: either directed or multigraph (default)
* `-s [SIZE]` / `--sample [SIZE]`: classifies a random sample of `SIZE` movies
* `-v` / `--verbose`: prints debugging and general graph information 
* `-o [FILE]` / `--output_predictions [FILE]`: prints each movie's predicted
random graph label in FILE in CSV format

For help information, use the -h or --help flag:
```
python driver.py -h
```

