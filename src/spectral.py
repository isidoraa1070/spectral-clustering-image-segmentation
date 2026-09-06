import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.linalg import eigh


def build_similarity_matrix(X, sigma=1.0):
    """
    Builds similarity matrix using the Gaussian (RBF) kernel.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data points.
    sigma : float
        Controls how quickly similarity decays with distance.
        Smaller sigma -> only very close points are considered similar.

    Returns
    -------
    W : np.ndarray, shape (n_samples, n_samples)
        Symmetric similarity matrix, W[i, j] in (0, 1].
    """
    # Pairwise squared Euclidean distances between all points in X
    dist_sq = cdist(X, X, metric='sqeuclidean')

    # Gaussian (RBF) kernel: converts distance into similarity
    W = np.exp(-dist_sq / (2 * sigma**2))

    return W


def compute_laplacian(W, normalized=True):
    """
    Computes the graph Laplacian from a similarity matrix.

    Parameters
    ----------
    W : np.ndarray, shape (n_samples, n_samples)
        Similarity matrix.
    normalized : bool
        If True, returns the symmetric normalized Laplacian L_sym.
        If False, returns the unnormalized Laplacian L.

    Returns
    -------
    L : np.ndarray, shape (n_samples, n_samples)
        Graph Laplacian.
    """
    # Degree matrix: D[i, i] = total connection strength of point i to all others
    D = np.diag(np.sum(W, axis=1))

    # Unnormalized Laplacian
    L = D - W

    if not normalized:
        return L

    # Symmetric normalized Laplacian (Ng-Jordan-Weiss): L_sym = D^(-1/2) L D^(-1/2)
    # Balances the influence of points with very different degrees
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
    L_sym = np.matmul(np.matmul(D_inv_sqrt, L), D_inv_sqrt)

    return L_sym


def spectral_clustering(X, k, sigma=1.0, normalized=True):
    """
    Performs spectral clustering on input data X.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data points.
    k : int
        Number of clusters.
    sigma : float
        Bandwidth parameter for the Gaussian kernel.
    normalized : bool
        Whether to use the normalized Laplacian.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Cluster assignment for each point.
    """

    W = build_similarity_matrix(X, sigma=sigma)

    
    L = compute_laplacian(W, normalized=normalized)


    _, eigenvectors = eigh(L)

    # Keep the k eigenvectors corresponding to
    # the k smallest eigenvalues
    U = eigenvectors[:, :k]

    # For the normalized Laplacian, normalize each row
    # of U to unit length before applying K-means
    if normalized:
        row_norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.maximum(row_norms, 1e-12)

    # Apply K-means in the spectral embedding space
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(U)
    labels = kmeans.labels_

    return labels

def spectral_clustering_from_similarity(W, k, normalized=True):
    """
    Performs spectral clustering given a precomputed similarity matrix.

    Parameters
    ----------
    W : np.ndarray, shape (n_samples, n_samples)
        Precomputed similarity matrix.
    k : int
        Number of clusters.
    normalized : bool
        Whether to use the normalized Laplacian.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Cluster assignment for each point.
    """
    
    L = compute_laplacian(W, normalized=normalized)

    
    _, eigenvectors = eigh(L)


    U = eigenvectors[:, :k]

    # For the normalized Laplacian, normalize each row
    # of U to unit length before applying K-means
    if normalized:
        row_norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.maximum(row_norms, 1e-12)

    # Apply K-means in the spectral embedding space
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(U)
    labels = kmeans.labels_

    return labels
