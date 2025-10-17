from scipy.stats._multivariate import multivariate_normal_frozen,\
    matrix_normal_frozen
from scipy.stats import norm
import numpy as np


class norm_multivariate(multivariate_normal_frozen):

    @property
    def cov_sqrt(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues))

    @property
    def nbVar(self):
        return self.cov.shape[0]

    def ppf(self, randUnifArr):
        shape = randUnifArr.shape
        randUnifArray = randUnifArr.reshape(-1, self.nbVar)
        indepStdNormArr = norm.ppf(randUnifArray)
        corrNormArr = np.array([self.cov_sqrt @ sample
                                for sample in indepStdNormArr])
        corrNormArr += np.atleast_2d(self.mean)
        return corrNormArr.reshape(shape)


class norm_matrix(matrix_normal_frozen):

    @property
    def colcov_sqrt(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.colcov)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues))

    @property
    def nbVar(self):
        return self.colcov.shape[0]

    def ppf(self, randUnifArr):
        shape = randUnifArr.shape
        randUnifArray = randUnifArr.reshape(-1, self.nbVar)
        indepStdNormArr = norm.ppf(randUnifArray)
        corrNormArr = np.array([self.colcov_sqrt @ sample
                                for sample in indepStdNormArr])
        corrNormArr += np.atleast_2d(self.mean)
        return corrNormArr.reshape(shape)
