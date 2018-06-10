# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from scipy import linalg
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

##  ==========================================================================
##      reviser : Liang He
##   descrption : linear discriminant analysis
##                revised from sklearn
##      created : 20170104
##      revised : 20180602
## 
##    Liang He, +86-13426228839, heliang@mail.tsinghua.edu.cn 
##    Aurora Lab, Department of Electronic Engineering, Tsinghua University
##  ==========================================================================

__all__ = ['LinearDiscriminantAnalysis']

def _cov(X):
    """Estimate covariance matrix.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    Returns
    -------
    s : array, shape (n_features, n_features)
        Estimated covariance matrix.
    """
    s = np.cov(X, rowvar=0)
    return s


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like, shape (n_features,)
        Class means.
    """
    means = []
    classes = np.unique(y)
    for group in classes:
        Xg = X[y == group, :]
        means.append(Xg.mean(0))
    return np.asarray(means)

def _class_cov(X, y):
    """Compute class covariance matrix.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.
    shrinkage : string or float, optional
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.
    Returns
    -------
    cov : array-like, shape (n_features, n_features)
        Class covariance matrix.
    """
    classes = np.unique(y)
    covs = []
    for group in classes:
        Xg = X[y == group, :]
        covs.append(np.atleast_2d(_cov(Xg)))
    return np.average(covs, axis=0)


class LinearDiscriminantAnalysis:
                
    def __init__(self, n_components=None, within_between_ratio=10.0, 
                 nearest_neighbor_ratio=1.2):
        self.n_components = n_components
        self.within_between_ratio = within_between_ratio
        self.nearest_neighbor_ratio = nearest_neighbor_ratio
        
    def _solve_eigen(self, X, y):
        """Eigenvalue solver.
        The eigenvalue solver computes the optimal solution of the Rayleigh
        coefficient (basically the ratio of between class scatter to within
        class scatter). This solver supports both classification and
        dimensionality reduction (with optional shrinkage).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        Notes
        -----
        This solver is based on [1]_, section 3.8.3, pp. 121-124.
        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self.means_ = _class_means(X, y)        
        self.covariance_ = _class_cov(X, y)

        Sw = self.covariance_  # within scatter
        St = _cov(X)  # total scatter
        Sb = St - Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, Sw)        
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        evecs /= np.linalg.norm(evecs, axis=0)
        self.scalings_ = np.asarray(evecs)
        
                
    def fit(self, X, y):
        """Fit Local Pairwise Trained Linear Discriminant Analysis 
           model according to the given training data and parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array, shape (n_samples,)
            Target values.
        """
        
        X, y = check_X_y(np.asarray(X), np.asarray(y.reshape(-1)), ensure_min_samples=2)
        self.classes_ = unique_labels(y)
        
        # Get the maximum number of components
        if self.n_components is None:
            self.n_components = len(self.classes_) - 1
        else:
            self.n_components = min(len(self.classes_) - 1, self.n_components)
            
        self._solve_eigen(np.asarray(X), np.asarray(y))
        return self
    
    
    
    def transform(self, X):
        """Project data to maximize class separation.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, ['scalings_'], all_or_any=any)
        X = check_array(X)
        X_new = np.dot(X, self.scalings_)
        return X_new[:, :self.n_components]



if __name__ == '__main__':
    
    samples = 20
    dim = 6
    lda_dim = 3
    
    data = np.random.random((samples, dim))  
    label = np.random.random_integers(0, 2, size=(samples, 1))
    
    lda = LinearDiscriminantAnalysis(lda_dim)
    lda.fit(data, label)
    lda_data = lda.transform(data)
    
    print (lda_data)
    