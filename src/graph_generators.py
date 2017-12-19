# Random Graph Generators
# Each class (besides GraphModel) has a constructor that creates either 
# a directed simple graph or an undirected multigraph based on the original 
# character network graph. The class methods allow the user to print summary
# metrics of the generated graphs.

import snap
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Converts the name into a format for filenames
def underscore_name(name):
    return '_'.join(name.lower().split())

# Base class representing a graph, with methods to describe the graph
class GraphModel(object):
    def __init__(self, orig_graph):
        self.name = 'Graph'
    
    # Provides a brief overview of characteristics of the graph
    def summarize_metrics(self):
        print 'Name: {}'.format(self.name)
        print 'Number of Nodes: {}'.format(self.get_num_nodes())
        print 'Number of Edges: {}'.format(self.get_num_edges())
        print 'Filename of Indegree Distribution: {}'.format( \
            self.plot_indegree_dist())
        print 'Filename of Outdegree Distribution: {}'.format( \
            self.plot_outdegree_dist())
        print 'Average Cluster Coefficient: {}'.format( \
            self.get_clustering_coefficient())
        print ''

    # Returns the number of nodes in the graph
    def get_num_nodes(self):
        return self.graph.GetNodes()

    # Returns the number of edges in the graph
    def get_num_edges(self):
        return self.graph.GetEdges()

    # Helper method to plot a degree distribution and return 
    # the filename of that plot (in the bin directory)
    def plot_degree_dist(self, degrees, deg_type):
        frequencies = { deg: degrees.count(deg) for deg in degrees }
        x, y = zip(*(sorted(frequencies.items(), key=lambda x: x[0])))
        total_degrees = sum(y)
        y = [val / float(total_degrees) for val in y]
        plt.plot(x, y)
        plt.title('{} Histogram'.format(deg_type))
        plt.xlabel(deg_type)
        plt.ylabel('Proportion of Nodes with {}'.format(deg_type))
        filename = '../bin/{}_{}.png'.format( \
            underscore_name(self.name), underscore_name(deg_type))
        plt.savefig(filename)
        plt.close()
        return filename

    # Plots the indegree distribution and returns the filename of the plot
    def plot_indegree_dist(self):
        indegrees = []
        for node in self.graph.Nodes():
            indegrees.append(node.GetInDeg())
        return self.plot_degree_dist(indegrees, 'Indegree')

    # Plots the outdegree distribution and returns the filename of the plot
    def plot_outdegree_dist(self):
        outdegrees = []
        for node in self.graph.Nodes():
            outdegrees.append(node.GetOutDeg())
        return self.plot_degree_dist(outdegrees, 'Outdegree')

    # Returns the ratio of (# of edges connecting node i's neighbors) to
    # (# of possible edges connecting node i's neighbors) over all nodes i
    def get_clustering_coefficient(self):
        return snap.GetClustCf(self.graph)

    # Draws the graph -- implemented in the child classes
    def draw_graph(self, name=None):
        pass

# Represents a directed graph with methods to draw the graph with NetworkX
class DirectedGraphModel(GraphModel):
    def __init__(self, orig_graph):
        super(DirectedGraphModel, self).__init__(orig_graph)
        self.name = 'Directed Graph'
        self.reset_graph(orig_graph)
        for edge in orig_graph.Edges():
            self.graph.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
        self.create_nx_graph()

    # Sets self.graph to be a graph with the same nodes as orig_graph 
    # (which is a Snap graph object), but with zero edges
    def reset_graph(self, orig_graph):
        self.num_orig_nodes = orig_graph.GetNodes()
        self.num_orig_edges = orig_graph.GetEdges()
        self.graph = snap.PNGraph.New()
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())

    # Produces a NetworkX graph from self.graph and computes the node
    # coordinates for the NetworkX graph object (for plotting the graph)
    def create_nx_graph(self):
        self.nx_graph = nx.DiGraph()
        node_list = [node.GetId() for node in self.graph.Nodes()]
        self.nx_graph.add_nodes_from(node_list)
        edge_list = [(edge.GetSrcNId(), edge.GetDstNId()) \
            for edge in self.graph.Edges()]
        self.nx_graph.add_edges_from(edge_list)
        self.nx_pos = nx.spring_layout(self.nx_graph)

    # Draws the NetworkX graph, with an optional filename.
    # nx_pos is a dictionary mapping to the coordinates of the nodes
    # it's the pos parameter for networkx drawing functions
    def draw_graph(self, nx_pos, name=None):
        # Convert to networkx graph object
        nx.draw(self.nx_graph, nx_pos)
        file_suffix = name if name is not None else self.name
        filename = '../bin/{}.png'.format(file_suffix)
        plt.savefig(filename)
        plt.close()

# Based on the original graph, produces an Erdos-Renyi graph so that 
# every possible edge between nodes has probability E / (N * (N - 1))
# of existing (where E is the original number of edges and N is the 
# original number of nodes)
class DirectedErdosRenyi(DirectedGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Directed Erdos Renyi'
        self.reset_graph(orig_graph)
        n = self.num_orig_nodes
        # Note that there are n * (n - 1) possible directed edges
        prob_edge_exists = float(self.num_orig_edges) / (n * (n - 1))
        for node1 in orig_graph.Nodes():
            for node2 in orig_graph.Nodes():
                if node1.GetId() == node2.GetId():
                    continue
                if random.random() < prob_edge_exists:
                    self.graph.AddEdge(node1.GetId(), node2.GetId())
        self.create_nx_graph()

# Based on the original graph, produces a Chung Lu graph so that given node i
# with original outdegree D_i and node j with original indegree d_j, the 
# probability of edge i -> j existing is proportional to D_i * d_j
class DirectedChungLu(DirectedGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Directed Chung Lu'
        self.reset_graph(orig_graph)
        for node1 in orig_graph.Nodes():
            for node2 in orig_graph.Nodes():
                if node1.GetId() == node2.GetId():
                    continue
                proportional = node1.GetInDeg() * node2.GetOutDeg()
                if random.random() < proportional / float(self.num_orig_edges):
                    self.graph.AddEdge(node2.GetId(), node1.GetId())
        self.create_nx_graph()

# Based on the original graph, produces a graph from the configuration model
# In this process, we split the original graphs' edges into "stubs" (one 
# in-stub and one out-stub), which we then randomly pair to produce new edges 
# for the new graph (so every edge consists of one in-stub and one out-stub)     
class DirectedConfiguration(DirectedGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Directed Configuration'
        in_stubs = []
        out_stubs = []
        for node in orig_graph.Nodes():
            in_stubs += [node.GetId()] * node.GetInDeg()
            out_stubs += [node.GetId()] * node.GetOutDeg()
        graph_valid = False
        while not graph_valid:
            graph_valid = True
            self.reset_graph(orig_graph)
            random.shuffle(in_stubs) # only one list needs to be shuffled
            for i in range(len(in_stubs)):
                if in_stubs[i] == out_stubs[i]:
                    graph_valid = False
                    break
                # For time purposes, ignore multiple edges
                self.graph.AddEdge(out_stubs[i], in_stubs[i])
        self.create_nx_graph()

# Based on the original graph, produces a graph from the Fast Reciprocal 
# Directed model (see Durak et al. 2012). In implementation, the model is
# similar to the configuration model, but it first examines reciprocated 
# edges (a pair of nodes that have edges in both directions); we create 
# "stubs" of all reciprocated edges, pair the stubs randomly, and create
# reciprocated edges with the pairings. Then, we proceed as with the 
# directed configuration for the original unreciprocated edges
class FastReciprocalDirected(DirectedGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Directed FRD'
        self.reset_graph(orig_graph)
        in_stubs = []
        out_stubs = []
        recip_stubs = []
        for node in orig_graph.Nodes():
            in_stubs += [node.GetId()] * node.GetInDeg()
            out_stubs += [node.GetId()] * node.GetOutDeg()
            for nbr in node.GetOutEdges():
                if orig_graph.IsEdge(nbr, node.GetId()):
                    recip_stubs.append(node.GetId())
        num_recip_edges = len(recip_stubs) / 2
        num_single_edges = self.num_orig_edges - 2 * num_recip_edges
        for i in range(num_recip_edges):
            idx1 = random.randint(0, len(recip_stubs) - 1)
            idx2 = random.randint(0, len(recip_stubs) - 1)
            # Current Disposal process for self-loops and repeated edges:
            # just skip that edge
            if recip_stubs[idx1] == recip_stubs[idx2]:
                continue
            self.graph.AddEdge(recip_stubs[idx1], recip_stubs[idx2])
            self.graph.AddEdge(recip_stubs[idx2], recip_stubs[idx1])
        for i in range(num_single_edges):
            idx1 = random.randint(0, len(out_stubs) - 1)
            idx2 = random.randint(0, len(in_stubs) - 1)
            # Current Disposal process for self-loops and repeated edges:
            # just skip that edge
            if out_stubs[idx1] == in_stubs[idx2]:
                continue
            self.graph.AddEdge(out_stubs[idx1], in_stubs[idx2])
        self.create_nx_graph()

# Based on the original graph, produces a directed Preferential Attachment
# graph, in which we introduce nodes i into the new graph. With prob_uniform
# probability, we create an edge from i to a uniformly randomly chosen node.
# Otherwise, we randomly choose the other endpoint for the new edge with
# probabilities proportional to their current indegrees (the scheme is 
# described as "the rich get richer")
class DirectedPreferentialAttachment(DirectedGraphModel):
    def __init__(self, orig_graph, ordered=False, prob_uniform=0.2):
        self.name = 'Directed Preferential Attachment'
        self.reset_graph(orig_graph)
        node_ids = []
        for node in orig_graph.Nodes():
            node_ids.append(node.GetId())
        random.shuffle(node_ids)
        for node_id in node_ids:
            for i in range(orig_graph.GetNI(node_id).GetOutDeg()):
                cands = [cand for cand in node_ids if cand != node_id and \
                    not self.graph.IsEdge(node_id, cand)]
                weights = [self.graph.GetNI(cand).GetInDeg() for cand in cands]
                if random.random() < prob_uniform or sum(weights) == 0:
                    neighbor = cands[random.randint(0, len(cands) - 1)]
                else:
                    weights = [w / float(sum(weights)) for w in weights]
                    neighbor = np.random.choice(cands, p=weights)
                self.graph.AddEdge(node_id, neighbor)
        self.create_nx_graph()

# Represents an undirected multigraph with methods to draw with NetworkX
# Recall that while TNEANet yields directed graphs, we are only interested
# in undirected multigraphs in this case
class UndirectedMultiGraphModel(GraphModel):
    def __init__(self, orig_graph):
        super(UndirectedMultiGraphModel, self).__init__(orig_graph)
        self.name = 'Multigraph'
        self.reset_graph(orig_graph)
        for edge in orig_graph.Edges():
            self.graph.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
        self.create_nx_graph()

    # Sets self.graph to be a graph with the same nodes as orig_graph 
    # (which is a Snap graph object), but with zero edges
    def reset_graph(self, orig_graph):
        self.num_orig_nodes = orig_graph.GetNodes()
        # Half number of edges because we're interested in undirected edges
        self.num_orig_edges = orig_graph.GetEdges() / 2
        self.graph = snap.TNEANet.New()
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())

    # Produces a NetworkX graph from self.graph and computes the node
    # coordinates for the NetworkX graph object (for plotting the graph)
    def create_nx_graph(self):
        self.nx_graph = nx.MultiGraph()
        node_list = [node.GetId() for node in self.graph.Nodes()]
        self.nx_graph.add_nodes_from(node_list)
        edge_list = [(edge.GetSrcNId(), edge.GetDstNId()) \
            for edge in self.graph.Edges()]
        self.nx_graph.add_edges_from(edge_list)
        self.nx_pos = nx.spring_layout(self.nx_graph)

    # Draws the NetworkX graph, with an optional filename.
    # nx_pos is a dictionary mapping to the coordinates of the nodes
    # it's the pos parameter for networkx drawing functions
    def draw_graph(self, nx_pos, name=None):
        nx.draw_networkx_nodes(self.nx_graph, nx_pos)

        # Between two nodes, only one edge is drawn, but its width is 
        # proportional to the number of edges between the pair of nodes
        edge_list = sorted([(edge.GetSrcNId(), edge.GetDstNId()) \
            for edge in self.graph.Edges()])
        left_ptr = 0
        while left_ptr < len(edge_list):
            right_ptr = left_ptr + 1
            while right_ptr < len(edge_list) and \
                edge_list[right_ptr] == edge_list[left_ptr]:

                right_ptr += 1
            nx.draw_networkx_edges(self.nx_graph, nx_pos, \
                edgelist=[edge_list[left_ptr]], \
                width=float(right_ptr - left_ptr) / 2)
            left_ptr = right_ptr

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        file_suffix = name if name is not None else self.name
        filename = '../bin/{}.png'.format(file_suffix)
        plt.savefig(filename)
        plt.close()

# Based on the original graph, produces a graph from the configuration model
# In this process, we split the original graphs' edges into "stubs", which we 
# then randomly pair to produce new edges for the new graph 
class MultiConfiguration(UndirectedMultiGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Multigraph Configuration'
        self.reset_graph(orig_graph)
        stubs = []
        for node in orig_graph.Nodes():
            stubs += [node.GetId()] * node.GetInDeg()
        random.shuffle(stubs)
        for i in range(0, len(stubs), 2):
            if stubs[i] != stubs[i + 1]:
                self.graph.AddEdge(stubs[i], stubs[i + 1])
                self.graph.AddEdge(stubs[i + 1], stubs[i])
        self.create_nx_graph()

# Based on the original graph, produces an Erdos-Renyi null graph
# In this definition, we are guaranteed to have self.num_orig_edges edges
# in the new graph, which we randomly sample from a uniform probability
# distribution over possible edge locations
class MultiErdosRenyi(UndirectedMultiGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Multigraph Erdos Renyi'
        self.reset_graph(orig_graph)
        node_list = []
        for node in orig_graph.Nodes():
            node_list.append(node.GetId())
        for _ in range(self.num_orig_edges):
            idx1 = random.randint(0, len(node_list) - 1)
            idx2 = idx1
            while idx2 == idx1:
                idx2 = random.randint(0, len(node_list) - 1)
            self.graph.AddEdge(node_list[idx1], node_list[idx2])
            self.graph.AddEdge(node_list[idx2], node_list[idx1])
        self.create_nx_graph()

# Based on the original graph, produces a Chung Lu graph so that given nodes
# i and j with original degrees d_i and d_j, the probability of an edge (i, j)
# existing is proportional to d_i * d_j
class MultiChungLu(UndirectedMultiGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Multigraph Chung Lu'
        self.reset_graph(orig_graph)
        node_list = []
        deg_dist = []
        for node in orig_graph.Nodes():
            node_list.append(node.GetId())
            deg_dist.append(node.GetDeg())
        edge_list = []
        edge_probs = []
        for idx1 in range(len(node_list)):
            for idx2 in range(idx1 + 1, len(node_list)):
                edge_list.append((node_list[idx1], node_list[idx2]))
                edge_probs.append(deg_dist[idx1] * deg_dist[idx2])
        normalizer = float(sum(edge_probs))
        edge_probs = [x / normalizer for x in edge_probs]
        for _ in range(self.num_orig_edges):
            edge_idx = np.random.choice(len(edge_list), p=edge_probs)
            self.graph.AddEdge(edge_list[edge_idx][0], edge_list[edge_idx][1])
            self.graph.AddEdge(edge_list[edge_idx][1], edge_list[edge_idx][0])
        self.create_nx_graph()

# Based on the original graph, produces a Preferential Attachment graph, in 
# which we introduce nodes i into the new graph. With some probability, we 
# create an edge from i to a uniformly randomly chosen node. Otherwise, we 
# randomly choose the other endpoint for the new edge with probabilities 
# proportional to their current degrees (the scheme is described as "the 
# rich get richer")
class MultiPreferentialAttachment(UndirectedMultiGraphModel):
    def __init__(self, orig_graph, smoothing=0.5):
        self.name = 'Multigraph Preferential Attachment'
        self.reset_graph(orig_graph)
        node_weights = {}
        for node in orig_graph.Nodes():
            node_weights[node.GetId()] = smoothing
        node_sequence = []
        for _ in range(2 * self.num_orig_edges):
            node_list = node_weights.keys()
            normalizer = sum(node_weights.values())
            node_probs = [node_weights[node_id] / normalizer \
                for node_id in node_list]
            idx = np.random.choice(len(node_list), p=node_probs)
            node_sequence.append(node_list[idx])
            node_weights[node_list[idx]] += 1.0
        for i in range(self.num_orig_edges):
            self.graph.AddEdge(node_sequence[2 * i], node_sequence[2 * i + 1])
            self.graph.AddEdge(node_sequence[2 * i + 1], node_sequence[2 * i])
        self.create_nx_graph()
