import numpy as np
from MinTree import MinTree


def c2Score(M, rowSet, colSet, nodeSusp):
    suspTotal = nodeSusp[0][list(rowSet)].sum() + nodeSusp[1][list(colSet)].sum()
    return M[list(rowSet), :][:, list(colSet)].sum(axis=None) + suspTotal


def run_fraudar(M, colWeights, nodeSusp=None):
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
    for i in range(bestNumDeleted):
        if deleted[i][0] == 0:
            finalRowSet.remove(deleted[i][1])
        else:
            finalColSet.remove(deleted[i][1])
    return ((finalRowSet, finalColSet), bestAveScore)

