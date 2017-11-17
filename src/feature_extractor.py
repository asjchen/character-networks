# Feature Extractor 

import snap
import numpy as np

# TODO: this method does not give good accuracy
def get_k_profiles(graph, k=3):
    # Find all the possible profiles
    possible_profiles = []
    def construct_k_graph(curr_graph, edge_pair):
        if edge_pair[0] == k:
            curr_profile = []
            for i in range(k):
                curr_profile.append([curr_graph.GetNI(i).GetInDeg(), curr_graph.GetNI(i).GetOutDeg()])
            curr_profile = sorted(curr_profile)
            if curr_profile not in possible_profiles:
                possible_profiles.append(curr_profile)
        else:
            new_edge_pair = [edge_pair[0], edge_pair[1] + 1]
            if new_edge_pair[1] >= k:
                new_edge_pair[0] += 1
                new_edge_pair[1] -= k
            construct_k_graph(curr_graph, new_edge_pair)
            if edge_pair[0] != edge_pair[1]:
                curr_graph.AddEdge(edge_pair[0], edge_pair[1])
                construct_k_graph(curr_graph, new_edge_pair)
                curr_graph.DelEdge(edge_pair[0], edge_pair[1])
    k_graph = snap.PNGraph.New()
    for i in range(k):
        k_graph.AddNode(i)
    construct_k_graph(k_graph, [0, 1])
    profile_props = np.zeros((len(possible_profiles),))
    node_list = []
    for node in graph.Nodes():
        node_list.append(node.GetId())

    def gather_k_nodes(curr_nodes):
        if len(curr_nodes) == k:
            curr_profile = []
            for i in range(k):
                curr_profile.append([0, 0])
            for idx1 in range(len(curr_nodes)):
                for idx2 in range(len(curr_nodes)):
                    if graph.IsEdge(node_list[curr_nodes[idx1]], node_list[curr_nodes[idx2]]):
                        curr_profile[idx1][1] += 1
                        curr_profile[idx2][0] += 1
            curr_profile = sorted(curr_profile)
            profile_idx = possible_profiles.index(curr_profile)
            profile_props[profile_idx] += 1.0
        else:
            start = 0
            if len(curr_nodes) > 0:
                start = curr_nodes[-1]
            for idx in range(start, len(node_list) - k + 1 + len(curr_nodes)):
                new_curr_nodes = curr_nodes + [idx]
                gather_k_nodes(new_curr_nodes)

    gather_k_nodes([])
    return profile_props / np.sum(profile_props)


def get_directed_laplacian(graph):
    n = graph.GetNodes()
    laplacian = np.zeros((n, n))
    node_ids = []
    node_id_to_idx = {}
    for node in graph.Nodes():
        node_id_to_idx[node.GetId()] = len(node_ids)
        node_ids.append(node.GetId())
    for i in range(n):
        # if graph.GetNI(node_ids[i]).GetInDeg() > 0:
        if graph.GetNI(node_ids[i]).GetOutDeg() > 0:
            laplacian[i, i] = 1.0
    for edge in graph.Edges():
        src_idx = node_id_to_idx[edge.GetSrcNId()]
        dst_idx = node_id_to_idx[edge.GetDstNId()]
        # laplacian[dst_idx][src_idx] -= 1.0 / graph.GetNI(edge.GetDstNId()).GetInDeg()
        laplacian[src_idx][dst_idx] -= 1.0 / graph.GetNI(edge.GetSrcNId()).GetOutDeg()
    return laplacian

# Make eigenvalue distribution off of Laplacian matrix
def get_eigenvalue_distribution(graph, num_buckets=10):
    laplacian = get_directed_laplacian(graph)
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    hist, bins = np.histogram(eigenvalues.real, bins=num_buckets, range=(0.0, 2.0))
    return hist / float(np.sum(hist))
    
