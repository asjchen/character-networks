# Feature Extractor 

import snap
import numpy as np


# Measures Proportion of each type of directed triad
# def get_3_profiles(graph):
#     # In-degree, then out degree
#     possible_profiles = [
#         [[0, 0], [0, 0], [0, 0]],
#         [[0, 0], [0, 1], [1, 0]],
#         [[0, 2], [1, 0], [1, 0]],
#         [[0, 1], [1, 0], [1, 1]],


#         [[0, 1], [0, 1], [2, 0]],
#         [[1, 1], [1, 1], [1, 1]],
#         [[0, 2], [1, 1], [2, 0]],

#     ]
#     # lots of possibility b/c recriprocal edges


#     node_ids = []
#     for node in graph.Nodes():
#         node_ids.append(node.GetId())
#     node_ids = sorted(node_ids)
#     counts = np.zeros((len(possible_profiles),))
#     for i in range(len(node_ids)):
#         for j in range(i + 1, len(node_ids)):
#             for k in range(j + 1, len(node_ids)):
#                 curr_profile = [[0, 0], [0, 0], [0, 0]]
#                 curr_triad = [node_ids[i], node_ids[j], node_ids[k]]
#                 for idx1 in range(3):
#                     for idx2 in range(3):
#                         if graph.IsEdge(curr_triad[idx1], curr_triad[idx2]):
#                             curr_profile[idx1][1] += 1
#                             curr_profile[idx2][0] += 1
#                 curr_profile = sorted(curr_profile)
#                 for idx in range(len(possible_profiles)):
#                     if curr_profile == possible_profiles[idx]:
#                         counts[idx] += 1
#     counts /= np.sum(counts)
#     return counts

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
    
