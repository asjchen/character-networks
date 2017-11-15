# Feature Extractor 

import snap
import numpy as np


# Measures Proportion of each type of directed triad




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
def get_eigenvalue_distribution(graph, num_buckets=20):
    laplacian = get_directed_laplacian(graph)
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    hist, bins = np.histogram(eigenvalues.real, bins=num_buckets, range=(0.0, 2.0))
    return hist / float(np.sum(hist))
    
