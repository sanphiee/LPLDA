# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from scipy import linalg
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

##  ==========================================================================
##       author : Liang He
##   descrption : local pairwise linear discriminant analysis
##                revised from sklearn
##      created : 20180613
##      revised : 
## 
##    Liang He, +86-13426228839, heliang@mail.tsinghua.edu.cn 
##    Aurora Lab, Department of Electronic Engineering, Tsinghua University
##  ==========================================================================

__all__ = ['LocalPairwiseTrainedLinearDiscriminantAnalysis']

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
    s = np.cov(X, rowvar=0, bias = 1)    
    return s

def _similarity_function(mean_vec, vecs):

#    dot_kernel = np.array([np.dot(mean_vec, vecs) for i in range(0,len(vecs))])
#    return dot_kernel
    mean_vec_norm = mean_vec / np.sqrt(np.sum(mean_vec ** 2))
    vecs_norm = vecs / np.sqrt(np.sum(vecs ** 2, axis=1))[:, np.newaxis]
    cosine_kernel = np.array([np.dot(mean_vec_norm, vecs_norm[i]) for i in range(len(vecs_norm))])
    return cosine_kernel
	
def _class_means_and_neighbor_means(X, y, k1, k2):
    """Compute class means and neighor means
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.
    k1: within_between_ratio
    k2: nearest_neighbor_ratio
    Returns
    -------
    means : array-like, shape (n_features,)
        Class means and neighbor means
    """
    means = []
    neighbor_means = []
    
    classes = np.unique(y)
    samples = np.size(y)
    
    for group in classes:
        Xg = X[y == group, :]
        Xg_count = Xg.shape[0]
        Xg_mean = Xg.mean(0)
        Xn = X[y != group, :]
        Xg_similarity = _similarity_function(Xg_mean, Xg)
        Xg_similarity_min = min(Xg_similarity)
        Xn_similarity = _similarity_function(Xg_mean, Xn)
        Xn_neighbor_count = len(Xn_similarity[Xn_similarity > Xg_similarity_min])
        Xn_neighbor_count = int(max(k1 * Xg_count, k2 * Xn_neighbor_count))
        Xn_neighbor_count = min(Xn_neighbor_count, samples - Xg_count)
        Xn_label = np.argsort(Xn_similarity)
        Xn_label = Xn_label[::-1]
        Xg_neighbor = np.array([Xn[Xn_label[i]] for i in range(Xn_neighbor_count)])
        Xg_neighbor_mean = Xg_neighbor.mean(0)
        
        means.append(Xg_mean)
        neighbor_means.append(Xg_neighbor_mean)
        
    return np.array(means), np.array(neighbor_means)

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

def _local_pairwise_cov(class_mean, neighbor_mean):
    """Estimate local pairwise matrix.
    Parameters
    ----------
    class_mean : array-like, shape (n_samples, n_features)
				 each class mean
    neighbor_mean: array-like, shape (n_samples, n_features)
				 each class neighbor mean
    Returns
    -------
    s : array, shape (n_features, n_features)
        Estimated covariance matrix.
    """
    covs = []
    for i in range(0, len(class_mean)):
        local_pair = np.vstack((class_mean[i], neighbor_mean[i]))
        covs.append(np.atleast_2d(_cov(local_pair)))
    return np.average(covs, axis=0)

class LocalPairwiseLinearDiscriminantAnalysis:
                
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
        self.means_, self.neighbor_means_ = _class_means_and_neighbor_means(
                X, y, self.within_between_ratio, self.nearest_neighbor_ratio)
    
        Sw = _class_cov(X, y) # within class cov
        Sb = _local_pairwise_cov(self.means_, self.neighbor_means_)
        
        evals, evecs = linalg.eigh(Sb, Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
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
            
        self._solve_eigen(X, y)
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
    
    lda = LocalPairwiseLinearDiscriminantAnalysis(lda_dim)
    lda.fit(data, label)
    lda_data = lda.transform(data)
    
    print (lda_data)
    