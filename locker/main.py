from sklearn.neighbors import BallTree
import numpy as np
from scipy.stats import multivariate_normal


class locker:
    def __init__(self, n_neighbors=5, leaf_size=40):
        '''
        Local Covariance Kernel Regression
        '''
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        self.neighbours_tree = BallTree(X, leaf_size=self.leaf_size)
        self.training_X = X
        self.training_y = y

    def predict(self, X):
        results_array = []
        uncertainties_array = []
        for x in X:
            # First find the nearby points
            dist, ind = self.neighbours_tree.query([x], k=self.n_neighbors)
            closest_x = self.training_X[ind[0], :]
            closest_y = self.training_y[ind[0]]

            # Now calculate covariance matrix of the inputs,
            # and also the inputs/outputs
            covariance_matrix = np.cov(closest_x, y=closest_y, rowvar=False)
            Cxx = covariance_matrix[:-1, :-1]
            Cxy = covariance_matrix[-1, :-1]
            Cyy = covariance_matrix[-1, -1]

            try:
                inverse_Cxx = np.linalg.inv(Cxx)
            except np.linalg.LinAlgError:
                raise ValueError('Covariance matrix is singular')

            # xy Element of inverse C matrix
            Vxy = np.matmul(Cxy, inverse_Cxx)

            # 1/yy element of inverse C matrix: schur complement
            s = Cyy - np.dot(Vxy, Cxy)

            # Probabilities of each value (distance weighting)
            kernel_densities = [
                multivariate_normal(x_value, Cxx) for x_value in closest_x
            ]
            numerator = np.sum(
                [
                    px.pdf(x) * (y_value + np.dot(Vxy, x - x_value))
                    for px, x_value, y_value in zip(
                        kernel_densities, closest_x, closest_y
                    )
                ]
            )
            denominator = np.sum([px.pdf(x) for px in kernel_densities])
            results_array.append(numerator / denominator)

            uncertainties_array.append(np.sqrt(s))
        return np.array(results_array), np.array(uncertainties_array)
