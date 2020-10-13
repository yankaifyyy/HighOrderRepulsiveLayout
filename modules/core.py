import theano
import theano.tensor as T


# Squared Euclidean distance between all pairs of row-vectors
# 这里X表示N×D数据矩阵，其中行对应于N个实例，列对应于D维。
# 这里低维距离矩阵，可以用到树加速
def sqeuclidean_var(X):
    N = X.shape[0]
    ss = (X ** 2).sum(axis=1)   # 先对点求平方，再对行求和
    # 返回点距离平方矩阵
    return ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * X.dot(X.T)


# Conditional probabilities of picking (ordered) pairs in high-dim space.
# 这里X表示归一化的距离矩阵或者邻接矩阵
def p_ij_conditional_var(X, sigma):

    N = X.shape[0]

    sqdistance = X**2

    # 每一行除以固定sigma，因为每一行对应固定i，每一列对应固定j
    esqdistance = T.exp(-sqdistance / ((2 * (sigma**2)).reshape((N, 1))))
    esqdistance_zd = T.fill_diagonal(
        esqdistance, 0)    # 原本对角线为1，填充为0，因为同一个点概率定义为0

    row_sum = T.sum(esqdistance_zd, axis=1).reshape(
        (N, 1))     # 对每一行求和，得到一行，reshape成一列，这样每一行才能除以不同数

    return esqdistance_zd / row_sum  # Possibly dangerous  返回一个矩阵


# Probabilities of picking pairs in low-dim space (using Student
# t-distribution).
def q_ij_student_t_var(Y):

    sqdistance = sqeuclidean_var(Y)
    one_over = T.fill_diagonal(1 / (sqdistance + 1), 0)
    return one_over / one_over.sum()
