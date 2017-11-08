# Null Directed Model Generators

import snap
import random
import matplotlib.pyplot as plt

# TODO: for each generator, have a constructor method that takes in the number of nodes and edges
# also has checker methods that output properties (for debugging mostly)

def underscore_name(name):
    return '_'.join(name.lower().split())

class DirectedGraphModel:
    def __init__(self, orig_graph):
        self.name = 'Graph'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges()
        self.graph = snap.PNGraph.New()
    
    def summarize_metrics(self):
        print 'Name: {}'.format(self.name)
        print 'Number of Nodes: {}'.format(self.get_num_nodes())
        print 'Number of Edges: {}'.format(self.get_num_edges())
        print 'Filename of Indegree Distribution: {}'.format(self.plot_indegree_dist())
        print 'Filename of Outdegree Distribution: {}'.format(self.plot_outdegree_dist())
        print 'Average Cluster Coefficient: {}'.format(self.get_clustering_coefficient())
        print ''

    def get_num_nodes(self):
        return self.graph.GetNodes()

    def get_num_edges(self):
        return self.graph.GetEdges()

    def plot_degree_dist(self, degrees):
        frequencies = { deg: degrees.count(deg) for deg in degrees }
        x, y = zip(*(sorted(frequencies.items(), key=lambda x: x[0])))
        total_degrees = sum(y)
        y = [val / float(total_degrees) for val in y]
        plt.plot(x, y)

    def plot_indegree_dist(self):
        indegrees = []
        for node in self.graph.Nodes():
            indegrees.append(node.GetInDeg())
        self.plot_degree_dist(indegrees)
        plt.title('Indegree Histogram')
        plt.xlabel('Indegree')
        plt.ylabel('Proportion of Nodes with Indegree')
        filename = '../bin/{}_in.png'.format(underscore_name(self.name))
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_outdegree_dist(self):
        outdegrees = []
        for node in self.graph.Nodes():
            outdegrees.append(node.GetOutDeg())
        self.plot_degree_dist(outdegrees)
        plt.title('Outdegree Histogram')
        plt.xlabel('Outdegree')
        plt.ylabel('Proportion of Nodes with Outdegree')
        filename = '../bin/{}_out.png'.format(underscore_name(self.name))
        plt.savefig(filename)
        plt.close()
        return filename

    def get_clustering_coefficient(self):
        return snap.GetClustCf(self.graph)


    # average shortest path
    # size of largest connected component


class DirectedErdosRenyi(DirectedGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Erdos Renyi'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges()
        self.graph = snap.PNGraph.New()
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())
        prob_edge_exists = float(num_edges) / (num_nodes * (num_nodes - 1))
        for node1 in orig_graph.Nodes():
            for node2 in orig_graph.Nodes():
                if node1.GetId() == node2.GetId():
                    continue
                if random.random() < prob_edge_exists:
                    self.graph.AddEdge(node1.GetId(), node2.GetId())


class DirectedChungLu(DirectedGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Chung Lu'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges()
        self.graph = snap.PNGraph.New()
        sum_indegrees = 0.0
        sum_outdegrees = 0.0
        sum_doubled_degs = 0.0
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())
            sum_indegrees += node.GetInDeg()
            sum_outdegrees += node.GetOutDeg()
            sum_doubled_degs += node.GetInDeg() * node.GetOutDeg()
        normalizer = (sum_indegrees * sum_outdegrees - sum_doubled_degs) / float(num_edges)
        for node1 in orig_graph.Nodes():
            for node2 in orig_graph.Nodes():
                if node1.GetId() == node2.GetId():
                    continue
                if random.random() < node1.GetInDeg() * node2.GetOutDeg() / normalizer:
                    self.graph.AddEdge(node1.GetId(), node2.GetId())


example = snap.PNGraph.New()
for i in range(1000):
    example.AddNode(i)
for i in range(1000):
    for j in range(1000):
        if random.random() < 0.25:
            example.AddEdge(i, j)
erdos_renyi = DirectedErdosRenyi(example)
erdos_renyi.summarize_metrics()
chung_lu = DirectedChungLu(example)
chung_lu.summarize_metrics()



