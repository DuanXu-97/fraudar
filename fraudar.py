import numpy as np
from MinTree import MinTree


def c2Score(M, rowSet, colSet, nodeSusp):
    suspTotal = nodeSusp[0][list(rowSet)].sum() + nodeSusp[1][list(colSet)].sum()
    return M[list(rowSet), :][:, list(colSet)].sum(axis=None) + suspTotal


def run_fraudar(M, dm, numToDetect):
    Mcur = M.copy().tolil()
    res = []
    for i in range(numToDetect):
        colWeights = dm.get_weights()
        weight_matrix = dm.get_weighted_matrix()
        ((rowSet, colSet), score) = GreedyDecreasing(weight_matrix, colWeights, userDeleNum=684556//numToDetect+1, objDeleNum=85339//numToDetect+1)
        res.append(((rowSet, colSet), score))
        (rs, cs) = Mcur.nonzero()
        for i in range(len(rs)):
            if rs[i] in rowSet and cs[i] in colSet:
                Mcur[rs[i], cs[i]] = 0
        dm.update_matrix(Mcur)
        dm.update_weighted_matrix()
    return res

# 686556, 85539
def GreedyDecreasing(M, colWeights, userDeleNum, objDeleNum, nodeSusp=None):
    print(userDeleNum)
    print(objDeleNum)
    (m, n) = M.shape
    if nodeSusp is None:
        nodeSusp = (np.zeros(m), np.zeros(n))
    Md = M.todok()
    Ml = M.tolil()
    Mlt = M.transpose().tolil()
    rowSet = set(range(0, m))
    colSet = set(range(0, n))
    curScore = c2Score(M, rowSet, colSet, nodeSusp)

    bestAveScore = curScore / (len(rowSet) + len(colSet))
    bestSets = (rowSet, colSet)
    rowDeltas = np.squeeze(M.sum(axis=1).A) + nodeSusp[
        0]  # contribution of this row to total weight, i.e. *decrease* in total weight when *removing* this row
    colDeltas = np.squeeze(M.sum(axis=0).A) + nodeSusp[1]
    rowTree = MinTree(rowDeltas)
    colTree = MinTree(colDeltas)

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0

    while rowSet and colSet:
        if (len(colSet) + len(rowSet)) % 100000 == 0:
            print(("current set size = %d" % (len(colSet) + len(rowSet),)))
        (nextRow, rowDelt) = rowTree.getMin()
        (nextCol, colDelt) = colTree.getMin()
        if rowDelt <= colDelt:
            curScore -= rowDelt
            for j in Ml.rows[nextRow]:
                delt = colWeights[j]
                colTree.changeVal(j, -colWeights[j])
            rowSet -= {nextRow}
            rowTree.changeVal(nextRow, float('inf'))
            deleted.append((0, nextRow))
        else:
            curScore -= colDelt
            for i in Mlt.rows[nextCol]:
                delt = colWeights[nextCol]
                rowTree.changeVal(i, -colWeights[nextCol])
            colSet -= {nextCol}
            colTree.changeVal(nextCol, float('inf'))
            deleted.append((1, nextCol))

        numDeleted += 1
        curAveScore = curScore / (len(colSet) + len(rowSet))

        if curAveScore > bestAveScore:
            bestAveScore = curAveScore
            bestNumDeleted = numDeleted

    # reconstruct the best row and column sets
    finalRowSet = set(range(m))
    finalColSet = set(range(n))

    # 由于已知欺诈用户和物体的数量，因此可以依次校正删除的节点数量
    for i in range(userDeleNum):
        if deleted[i][0] == 0:
            finalRowSet.remove(deleted[i][1])
    for i in range(objDeleNum):
        if deleted[i][0] == 1:
            finalColSet.remove(deleted[i][1])

    # for i in range(bestNumDeleted):
    #     if deleted[i][0] == 0:
    #         finalRowSet.remove(deleted[i][1])
    #     else:
    #         finalColSet.remove(deleted[i][1])

    return ((finalRowSet, finalColSet), bestAveScore)

