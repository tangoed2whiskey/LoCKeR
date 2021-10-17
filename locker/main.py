import sklearn


class locker:
    def __init__(self, n_neighbors):
        '''
        Local Covariance Kernel Regression
        '''
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        ...

    def transform(self, X):
        ...

    def fit_transform(self, X, y):
        ...