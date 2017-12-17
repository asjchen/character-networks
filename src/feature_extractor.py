# Feature Extractor 

import snap
import numpy as np

# TODO: this method does not give good accuracy
# Only for unweighted directed graphs
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
            start = -1
            if len(curr_nodes) > 0:
                start = curr_nodes[-1]
            for idx in range(start + 1, len(node_list) - k + 1 + len(curr_nodes)):
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
def get_directed_eigenvalue_distribution(graph, num_buckets=20):
    laplacian = get_directed_laplacian(graph)
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    hist, bins = np.histogram(eigenvalues.real, bins=num_buckets, range=(0.0, 2.0))
    return hist / float(np.sum(hist))
    
def combine_directed_eigen_profiles(graph):
    eigen_features = get_directed_eigenvalue_distribution(graph)
    profile_features = get_k_profiles(graph)
    return np.concatenate((eigen_features, profile_features), axis=0)

def get_unnormalized_laplacian(graph):
    n = graph.GetNodes()
    laplacian = np.zeros((n, n))
    node_ids = []
    node_id_to_idx = {}
    for node in graph.Nodes():
        node_id_to_idx[node.GetId()] = len(node_ids)
        node_ids.append(node.GetId())
    for i in range(n):
        laplacian[i, i] = float(graph.GetNI(node_ids[i]).GetOutDeg())
    for edge in graph.Edges():
        src_idx = node_id_to_idx[edge.GetSrcNId()]
        dst_idx = node_id_to_idx[edge.GetDstNId()]
        laplacian[src_idx][dst_idx] -= 1.0
    return laplacian

def get_multi_eigenvalue_distribution(graph, num_buckets=20):
    laplacian = get_unnormalized_laplacian(graph)
    for i in range(laplacian.shape[0]):
        if laplacian[i, i] != 0:
            laplacian[i, :] /= laplacian[i, i]
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    hist, bins = np.histogram(eigenvalues.real, bins=num_buckets, range=(0.0, 2.0))
    return hist / float(np.sum(hist))

def get_histogram(values, num_buckets=10):
    if len(values) == 0:
        return np.full((num_buckets,), 0.0)
    min_value = min(values)
    max_value = max(values)
    hist = np.full((num_buckets,), 1.0)
    if min_value != max_value:
        normed_values = [float(x - min_value) / (max_value - min_value) for x in values]
        hist, bins = np.histogram(normed_values, bins=num_buckets, range=(0.0, 1.0))
    return hist / float(np.sum(hist))

def get_betweenness_centrality_dist(graph, num_buckets=10):
    node_cent = snap.TIntFltH()
    edge_cent = snap.TIntPrFltH()
    snap.GetBetweennessCentr(graph, node_cent, edge_cent, 1.0)
    node_betweenness = []
    edge_betweenness = []
    for node in node_cent:
        node_betweenness.append(node_cent[node])
    for edge in edge_cent:
        edge_betweenness.append(edge_cent[edge])
    node_hist = get_histogram(node_betweenness, num_buckets=num_buckets)
    edge_hist = get_histogram(edge_betweenness, num_buckets=num_buckets)
    return np.concatenate((node_hist, edge_hist), axis=0)

def get_node_centrality_stats(graph):
    cluster_coeff = snap.GetClustCf(graph)
    min_ecc = graph.GetNodes() + 1
    max_ecc = 0
    for node in graph.Nodes():
        ecc = snap.GetNodeEcc(graph, node.GetId())
        min_ecc = min(ecc, min_ecc)
        max_ecc = max(ecc, max_ecc)
    return np.array([cluster_coeff, max_ecc, min_ecc])

def get_pagerank_dist(graph):
    node_pagerank = snap.TIntFltH()
    snap.GetPageRank(graph, node_pagerank)
    return get_histogram([node_pagerank[node] for node in node_pagerank])

def get_multigraph_features(graph, num_cent_buckets=10, num_eigen_buckets=20):
    between_features = get_betweenness_centrality_dist(graph, num_buckets=num_cent_buckets)
    eigen_features = get_multi_eigenvalue_distribution(graph, num_buckets=num_eigen_buckets)
    centrality_features = get_node_centrality_stats(graph)
    pagerank_features = get_pagerank_dist(graph)
    return np.concatenate((between_features, eigen_features), axis=0)


