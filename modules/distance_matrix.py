import graph_tool.all as gt
import numpy as np
import itertools

num_x = 0.5
# 得到修改的邻接矩阵，相连的点矩阵值为权重的倒数，类似于距离，不相连的点距离设置为一个非常大的数


def get_adjacency_matrix(g):
    adj = gt.adjacency(g)
    X = np.zeros(adj.shape, dtype=np.float64)
    for i, j in zip(*adj.nonzero()):
        if(adj[i, j] == 1):
            X[i, j] = adj[i, j]
    return X


def get_modified_adjacency_matrix(g, k):
    # Get regular adjacency matrix
    adj = gt.adjacency(g)

    # Initialize the modified adjacency matrix
    X = np.zeros(adj.shape)

    # Loop over nonzero elements adj.nonzero()返回一个tuple ((非零元素行),(非零元素列))，前面加上*变成两个tuple
    for i, j in zip(*adj.nonzero()):
        X[i, j] = 1 / adj[i, j]

    adj_max = adj.max()

    # Loop over zero elements 循环零元素,difference是前边减后边,itertools.product进行全排列，返回元组
    for i, j in set(itertools.product(range(adj.shape[0]), range(adj.shape[1]))).difference(zip(*adj.nonzero())):
        X[i, j] = k * adj_max
       # print(X[i, j])
    return X


def get_shortest_path_distance_matrix(g, k=10, weights=None):
    # Used to find which vertices are not connected. This has to be this weird,
    # since graph_tool uses maxint for the shortest path distance between
    # unconnected vertices.
    def get_unconnected_distance():
        g_mock = gt.Graph()
        g_mock.add_vertex(2)
        shortest_distances_mock = gt.shortest_distance(g_mock)
        # 对于未连接的点返回int32_t最大值2147483647
        unconnected_dist = shortest_distances_mock[0][1]
        return unconnected_dist

    # Get the value (usually maxint) that graph_tool uses for distances between
    # unconnected vertices.
    unconnected_dist = get_unconnected_distance()  # 2147483647

    # Get shortest distances for all pairs of vertices in a NumPy array.
    X = gt.shortest_distance(g, weights=weights).get_2d_array(
        range(g.num_vertices()))

    # X == unconnected_dist 返回一个 真假矩阵。 X[X == unconnected_dist]返回一个向量
    if len(X[X == unconnected_dist]) > 0:
        print('[distance_matrix] There were disconnected components!')

    # Get maximum shortest-path distance (ignoring maxint)
    X_max = X[X != unconnected_dist].max()

    # Set the unconnected distances to k times the maximum of the other
    # distances.
    X[X == unconnected_dist] = k * X_max

    return X


# Return the distance matrix of g, with the specified metric.
def get_distance_matrix(g, distance_metric, normalize=True, k=10.0, verbose=True, weights=None):
    if verbose:
        print('[distance_matrix] Computing distance matrix (metric: {0})'.format(
            distance_metric))

    if distance_metric == 'shortest_path' or distance_metric == 'spdm':
        X = get_shortest_path_distance_matrix(g, weights=weights)
    elif distance_metric == 'modified_adjacency' or distance_metric == 'mam':
        X = get_modified_adjacency_matrix(g, k)
    else:
        raise Exception('Unknown distance metric.')

    # Just to make sure, symmetrize the matrix.
    X = (X + X.T) / 2

    # Force diagonal to zero
    X[range(X.shape[0]), range(X.shape[1])] = 0

    # Normalize matrix s.t. max is 1.
    max_dis = np.max(X)
    adj = get_adjacency_matrix(g=g)

    if normalize:
        X /= np.max(X)
    if verbose:
        print('[distance_matrix] Done!')

    return X**num_x, max_dis**num_x, adj
