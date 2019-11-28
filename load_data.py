from scipy import sparse

# 加入权重
def load_data(file_path):
    edgesSource = []
    edgesDest = []
    edgeWeight = []
    with open(file_path) as f:
        for line in f:
            toks = line.split()
            edgesSource.append(int(toks[0]))
            edgesDest.append(int(toks[1]))
            edgeWeight.append(int(toks[2]))
    return listToSparseMatrix(edgesSource, edgesDest, edgeWeight)


def listToSparseMatrix(edgesSource, edgesDest, edgeWeight):
    m = max(edgesSource) + 1
    n = max(edgesDest) + 1
    M = sparse.coo_matrix((edgeWeight, (edgesSource, edgesDest)), shape=(m, n))
    M1 = M > 0
    return M1.astype('int')
