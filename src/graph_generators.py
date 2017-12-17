# Null Directed Model Generators

import snap
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# TODO: reduce redundant code in various constructors

def underscore_name(name):
    return '_'.join(name.lower().split())

class GraphModel(object):
    def __init__(self, orig_graph):
        self.name = 'Graph'
    
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


class DirectedGraphModel(GraphModel):
    def __init__(self, orig_graph):
        super(DirectedGraphModel, self).__init__(orig_graph)
        self.name = 'Directed Graph'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges()
        self.graph = snap.PNGraph.New()
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())
        for edge in orig_graph.Edges():
            self.graph.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
        self.create_nx_graph()

    def create_nx_graph(self):
        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from([node.GetId() for node in self.graph.Nodes()])
        self.nx_graph.add_edges_from([(edge.GetSrcNId(), edge.GetDstNId()) for edge in self.graph.Edges()])
        self.nx_pos = nx.spring_layout(self.nx_graph)

    # nx_pos is a dictionary mapping to the coordinates of the nodes
    # it's the pos parameter for networkx drawing functions
    def draw_graph(self, nx_pos, name=None):
        # Convert to networkx graph object
        nx.draw(self.nx_graph, nx_pos)
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
        self.create_nx_graph()


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
        self.create_nx_graph()
        
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
        self.create_nx_graph()

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
        self.create_nx_graph()

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
                    neighbor = cands[random.randint(0, len(cands) - 1)]
                else:
                    weights = [w / float(sum(weights)) for w in weights]
                    neighbor = np.random.choice(cands, p=weights)
                #print '{} {}'.format(node_id, neighbor)
                self.graph.AddEdge(node_id, neighbor)
        self.create_nx_graph()


class UndirectedMultiGraphModel(GraphModel):
    def __init__(self, orig_graph):
        super(UndirectedMultiGraphModel, self).__init__(orig_graph)
        self.name = 'Multigraph'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges()
        self.graph = snap.TNEANet.New()
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())
        for edge in orig_graph.Edges():
            self.graph.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
        self.create_nx_graph()

    def create_nx_graph(self):
        # Convert to networkx graph object
        self.nx_graph = nx.MultiGraph()
        self.nx_graph.add_nodes_from([node.GetId() for node in self.graph.Nodes()])
        self.nx_graph.add_edges_from([(edge.GetSrcNId(), edge.GetDstNId()) for edge in self.graph.Edges()])
        self.nx_pos = nx.spring_layout(self.nx_graph)

    def draw_graph(self, nx_pos, name=None):
        nx.draw_networkx_nodes(self.nx_graph, nx_pos)

        edge_list = sorted([(edge.GetSrcNId(), edge.GetDstNId()) for edge in self.graph.Edges()])
        left_ptr = 0
        while left_ptr < len(edge_list):
            right_ptr = left_ptr + 1
            while right_ptr < len(edge_list) and edge_list[right_ptr] == edge_list[left_ptr]:
                right_ptr += 1
            nx.draw_networkx_edges(self.nx_graph, nx_pos, edgelist=[edge_list[left_ptr]], width=float(right_ptr - left_ptr) / 2)
            left_ptr = right_ptr

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        filename = '../bin/{}.png'.format(name if name is not None else self.name)
        plt.savefig(filename)
        plt.close()


class MultiConfiguration(UndirectedMultiGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Multigraph Configuration'
        self.graph = snap.TNEANet.New()
        stubs = []
        for node in orig_graph.Nodes():
            stubs += [node.GetId()] * node.GetInDeg()
            self.graph.AddNode(node.GetId())
        random.shuffle(stubs)
        for i in range(0, len(stubs), 2):
            if stubs[i] != stubs[i + 1]:
                self.graph.AddEdge(stubs[i], stubs[i + 1])
                self.graph.AddEdge(stubs[i + 1], stubs[i])
        self.create_nx_graph()

# In this definition, we are guaranteed to have num_edges edges
# in the new graph, which we randomly sample from a probability
# distribution over possible edge locations
class MultiErdosRenyi(UndirectedMultiGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Multigraph Erdos Renyi'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges() / 2
        node_list = []
        self.graph = snap.TNEANet.New()
        for node in orig_graph.Nodes():
            node_list.append(node.GetId())
            self.graph.AddNode(node.GetId())
        for _ in range(num_edges):
            idx1 = random.randint(0, len(node_list) - 1)
            idx2 = idx1
            while idx2 == idx1:
                idx2 = random.randint(0, len(node_list) - 1)
            self.graph.AddEdge(node_list[idx1], node_list[idx2])
            self.graph.AddEdge(node_list[idx2], node_list[idx1])
        self.create_nx_graph()

# Recall that while TNEANet yields directed graphs, 
class MultiChungLu(UndirectedMultiGraphModel):
    def __init__(self, orig_graph):
        self.name = 'Multigraph Chung Lu'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges() / 2
        self.graph = snap.TNEANet.New()
        node_list = []
        deg_dist = []
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())
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
        for _ in range(num_edges):
            edge_idx = np.random.choice(len(edge_list), p=edge_probs)
            self.graph.AddEdge(edge_list[edge_idx][0], edge_list[edge_idx][1])
            self.graph.AddEdge(edge_list[edge_idx][1], edge_list[edge_idx][0])
        self.create_nx_graph()

class MultiPreferentialAttachment(UndirectedMultiGraphModel):
    def __init__(self, orig_graph, smoothing=0.5):
        self.name = 'Multigraph Preferential Attachment'
        num_nodes = orig_graph.GetNodes()
        num_edges = orig_graph.GetEdges() / 2
        self.graph = snap.TNEANet.New()
        node_weights = {}
        for node in orig_graph.Nodes():
            self.graph.AddNode(node.GetId())
            node_weights[node.GetId()] = smoothing
        node_sequence = []
        for _ in range(2 * num_edges):
            node_list = node_weights.keys()
            normalizer = sum(node_weights.values())
            node_probs = [node_weights[node_id] / normalizer for node_id in node_list]
            idx = np.random.choice(len(node_list), p=node_probs)
            node_sequence.append(node_list[idx])
            node_weights[node_list[idx]] += 1.0
        for i in range(num_edges):
            self.graph.AddEdge(node_sequence[2 * i], node_sequence[2 * i + 1])
            self.graph.AddEdge(node_sequence[2 * i + 1], node_sequence[2 * i])
        self.create_nx_graph()



