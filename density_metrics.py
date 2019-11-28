import numpy as np
from scipy import sparse


class DensityMetric:
    def __init__(self, matrix):
        self.matrix = matrix
        self.weights = None
        self.weighted_matrix = None
        self.update_weighted_matrix()

    def update_weighted_matrix(self):
        self.weighted_matrix = self.matrix
        self.weights = [1] * self.matrix.shape[1]

    def update_matrix(self, matrix):
        self.matrix = matrix

    def get_weighted_matrix(self):
        return self.weighted_matrix

    def get_weights(self):
        return self.weights
    
    def get_matrix(self):
        return self.matrix


class SqrtWeightedAveDegree(DensityMetric):
    def __init__(self, matrix, c):
        self.c = c
        super(SqrtWeightedAveDegree, self).__init__(matrix=matrix)

    def update_weighted_matrix(self):
        (m, n) = self.matrix.shape
        colSums = self.matrix.sum(axis=0)
        colWeights = 1.0 / np.sqrt(np.squeeze(colSums) + self.c)
        colDiag = sparse.lil_matrix((n, n))
        colDiag.setdiag(colWeights)
        self.weighted_matrix = self.matrix * colDiag
        self.weights = colWeights


class LogWeightedAveDegree(DensityMetric):
    def __init__(self, matrix, c):
        self.c = c
        super(LogWeightedAveDegree, self).__init__(matrix=matrix)

    def update_weighted_matrix(self):
        (m, n) = self.matrix.shape
        colSums = self.matrix.sum(axis=0)
        colWeights = np.squeeze(np.array(1.0 / np.log(np.squeeze(colSums) + self.c)))
        colDiag = sparse.lil_matrix((n, n))
        colDiag.setdiag(colWeights)
        self.weighted_matrix = self.matrix * colDiag
        self.weights = colWeights


class AveDegree(DensityMetric):
    def __int__(self, matrix):
        super(AveDegree, self).__init__(matrix=matrix)



