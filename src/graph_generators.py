# Null Directed Model Generators

import snap
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# TODO: for each generator, have a constructor method that takes in the number of nodes and edges
# also has checker methods that output properties (for debugging mostly)

def underscore_name(name):
    return '_'.join(name.lower().split())

class GraphModel(object):
    def __init__(self, orig_graph):
        self.name = 'Graph'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges()
        self.graph = snap.PNGraph.New()
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())
        for edge in orig_graph.Edges():
            self.graph.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
    
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

    def draw_graph(self, name=None):
        pass


    # average shortest path
    # size of largest connected component


class DirectedGraphModel(GraphModel):
    def __init__(self, orig_graph):
        super(DirectedGraphModel, self).__init__(orig_graph)
        self.name = 'Directed Graph'

    def draw_graph(self, name=None):
        # Convert to networkx graph object
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from([node.GetId() for node in self.graph.Nodes()])
        nx_graph.add_edges_from([(edge.GetSrcNId(), edge.GetDstNId()) for edge in self.graph.Edges()])
        nx.draw(nx_graph)
        filename = '../bin/{}.png'.format(name if name is not None else self.name)
        plt.savefig(filename)
        plt.close()

class DirectedErdosRenyi(DirectedGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Directed Erdos Renyi'
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
        self.name = 'Directed Chung Lu'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges()
        self.graph = snap.PNGraph.New()
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())
        for node1 in orig_graph.Nodes():
            for node2 in orig_graph.Nodes():
                if node1.GetId() == node2.GetId():
                    continue
                if random.random() < node1.GetInDeg() * node2.GetOutDeg() / float(num_edges):
                    self.graph.AddEdge(node2.GetId(), node1.GetId())
        
class DirectedConfiguration(DirectedGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Directed Configuration'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges()
        in_stubs = []
        out_stubs = []
        for node in orig_graph.Nodes():
            in_stubs += [node.GetId()] * node.GetInDeg()
            out_stubs += [node.GetId()] * node.GetOutDeg()
        graph_valid = False
        count = 0
        while not graph_valid:
            count += 1
            graph_valid = True
            self.graph = snap.PNGraph.New()
            for node in orig_graph.Nodes():
                self.graph.AddNode(node.GetId())
            random.shuffle(in_stubs) # only one list needs to be shuffled
            for i in range(len(in_stubs)):
                if in_stubs[i] == out_stubs[i] or self.graph.IsEdge(out_stubs[i], in_stubs[i]):
                    graph_valid = False
                    break
                self.graph.AddEdge(out_stubs[i], in_stubs[i])

class FastReciprocalDirected(DirectedGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Directed FRD'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges()
        self.graph = snap.PNGraph.New()
        in_stubs = []
        out_stubs = []
        recip_stubs = []
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())
            in_stubs += [node.GetId()] * node.GetInDeg()
            out_stubs += [node.GetId()] * node.GetOutDeg()
            for nbr in node.GetOutEdges():
                if orig_graph.IsEdge(nbr, node.GetId()):
                    recip_stubs.append(node.GetId())
        num_recip_edges = len(recip_stubs) / 2
        num_single_edges = num_edges - 2 * num_recip_edges
        #print '{} {} {}'.format(num_recip_edges, num_single_edges, num_edges)
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

class DirectedPreferentialAttachment(DirectedGraphModel):
    def __init__(self, orig_graph, ordered=False, prob_uniform=0.2):
        self.name = 'Directed Preferential Attachment'
        node_ids = []
        self.graph = snap.PNGraph.New()
        for node in orig_graph.Nodes():
            node_ids.append(node.GetId())
            self.graph.AddNode(node.GetId())
        random.shuffle(node_ids)
        for node_id in node_ids:
            for i in range(orig_graph.GetNI(node_id).GetOutDeg()):
                cands = [cand for cand in node_ids if cand != node_id and not self.graph.IsEdge(node_id, cand)]
                weights = [self.graph.GetNI(cand).GetInDeg() for cand in cands]
                if random.random() < prob_uniform or sum(weights) == 0:
                    neighbor = node_ids[random.randint(0, len(cands) - 1)]
                else:
                    weights = [w / float(sum(weights)) for w in weights]
                    neighbor = np.random.choice(cands, p=weights)
                self.graph.AddEdge(node_id, neighbor)


class UndirectedMultiGraphModel(GraphModel):
    def __init__(self, orig_graph):
        super(MultiUndirectedGraphModel, self).__init__(orig_graph)
        self.name = 'Undirected Multigraph'

    def draw_graph(self, name=None):
        # Convert to networkx graph object
        nx_graph = nx.MultiGraph()
        nx_graph.add_nodes_from([node.GetId() for node in self.graph.Nodes()])
        nx_graph.add_edges_from([(edge.GetSrcNId(), edge.GetDstNId()) for edge in self.graph.Edges()])
        nx.draw(nx_graph)
        filename = '../bin/{}.png'.format(name if name is not None else self.name)
        plt.savefig(filename)
        plt.close()
        

def main():
    example = snap.PNGraph.New()
    for i in range(10):
        example.AddNode(i)
    for i in range(10):
        for j in range(10):
            if random.random() < 0.25:
                example.AddEdge(i, j)
    # erdos_renyi = DirectedErdosRenyi(example)
    # erdos_renyi.summarize_metrics()
    # chung_lu = DirectedChungLu(example)
    # chung_lu.summarize_metrics()
    configuration = DirectedConfiguration(example)
    configuration.summarize_metrics()

if __name__ == '__main__':
    main()



