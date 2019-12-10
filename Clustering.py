import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle


not_include = ['whitesmoke', 'white', 'snow', 'mistyrose', 'seashell', 'peachpuff', 'linen', 'bisque', 'red', 'antiquewhite', 'oldlace', 'floralwhite', 'cornsilk', 'lemonchiffon', 'ivory',
               'beige', 'lightyellow', 'lightgoldenrodyellow', 'lawngreen', 'honeydew', 'palegreen', 'mintcream', 'azure', 'lightcyan', 'paleturquoise', 'powderblue', 'skyblue', 'aliceblue',
               'lightslategrey', 'lightslategray', 'slategrey', 'slategray', 'ghostwhite', 'lavender', 'dimgray', 'dimgrey', 'darkgrey', 'darkgray', 'lightgray', 'gainsboro', 'darkmagenta',
               'lavenderblush', 'pink']
color_dict = {i: c for (i, c) in enumerate([color for color in colors.CSS4_COLORS if color not in not_include])}


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """
    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.array([np.cos(t) + 0.1 * np.random.randn(length), np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.array([2 * np.cos(t) + 0.1 * np.random.randn(length), 2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.array([3 * np.cos(t) + 0.1 * np.random.randn(length), 3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.array([4 * np.cos(t) + 0.1 * np.random.randn(length), 4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    plt.plot(circles[0, :], circles[1, :], '.k')
    plt.show()


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.
    :param path: the path to the APML_pic.pickle file
    """
    with open(path, 'rb') as f:
        apml = pickle.load(f)
    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle', genes_path='microarray_genes.pickle', conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.
    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5, 5], [-5, 5], 'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    distances = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        distances[i] = np.linalg.norm(Y - X[i], axis=1)
    return distances


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    center_of_mass = np.average(X, axis=0)
    return X[np.argmin(np.linalg.norm(X - center_of_mass, axis=1))]


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    centroids = np.array([X[np.random.choice(X.shape[0])]])
    distances = metric(X, centroids)
    distribution = distances.min(axis=1) / distances.sum(axis=1)
    distribution = np.nan_to_num(distribution)
    distribution /= distribution.sum()
    for _ in range(k-1):
        centroids = np.append(centroids, X[np.random.choice(X.shape[0], p=distribution)]).reshape((centroids.shape[0] + 1, centroids.shape[1]))
        distances = metric(X, centroids)
        distribution = distances.min(axis=1) / distances.sum(axis=1)
        distribution = np.nan_to_num(distribution)
        distribution /= distribution.sum()
    return centroids


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :return: a tuple of (clustering, centroids)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    centroids = init(X, k, metric)
    clusters = np.zeros(X.shape[0])
    for _ in range(iterations):
        distance_matrix = metric(X, centroids)
        clusters = distance_matrix.argmin(axis=1)
        for i in range(centroids.shape[0]):
            centroids[i] = center(X[np.argwhere(clusters == i)[:, 0]])
    return clusters, centroids


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """
    return np.exp(-(X ** 2) / (2 * (sigma ** 2)))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    distances = euclid(X, X)
    nearest_neighbors_inds = distances.argsort(axis=1)[:, : (m + 1)]  # this includes the point X[i] itself, thus for the m nearest neighbors, we use the indices in places {1,...,m}
    similarity_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        similarity_matrix[nearest_neighbors_inds[i], i] = 1
        similarity_matrix[i, nearest_neighbors_inds[i]] = 1
    return similarity_matrix


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    distance_matrix = euclid(X, X)
    similarity_matrix = similarity(distance_matrix, similarity_param)
    row_sums = similarity_matrix.sum(axis=1)
    inv_root_diagonal_degree_matrix = np.diag(row_sums ** -0.5)
    laplacian_matrix = np.eye(similarity_matrix.shape[0]) - np.dot(inv_root_diagonal_degree_matrix, np.dot(similarity_matrix, inv_root_diagonal_degree_matrix))
    eig_values, eig_vectors = np.linalg.eig(laplacian_matrix)
    eig_values, eig_vectors = np.real(eig_values), np.real(eig_vectors)
    k_smallest_eigenvalues_inds = np.argsort(eig_values)[: k]
    k_eigenvectors = eig_vectors[:, k_smallest_eigenvalues_inds]
    normalized_k_eigenvactors = k_eigenvectors / np.linalg.norm(k_eigenvectors, axis=1).reshape((k_eigenvectors.shape[0], 1))  # projecting rows onto the unit sphere
    normalized_k_eigenvactors = np.nan_to_num(normalized_k_eigenvactors)
    return kmeans(normalized_k_eigenvactors, k, 20)


def generate_circles():
    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.array([np.cos(t) + 0.1 * np.random.randn(length), np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.array([2 * np.cos(t) + 0.1 * np.random.randn(length), 2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.array([3 * np.cos(t) + 0.1 * np.random.randn(length), 3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.array([4 * np.cos(t) + 0.1 * np.random.randn(length), 4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    return circles.T


def get_data(circles=True):
    if circles:
        return generate_circles()
    else:
        with open('APML_pic.pickle', 'rb') as f:
            apml = pickle.load(f)
        return apml


def run_clustering():
    X = get_data()
    num_clusters = 16
    clusters, centroids = kmeans(X, num_clusters, 25)
    clusters_pts = [X[np.argwhere(clusters == i)[:, 0]] for i in range(num_clusters)]
    for i in range(len(clusters_pts)):
        pts = clusters_pts[i]
        if len(pts):
            plt.scatter(pts[:, 0], pts[:, 1], color=color_dict[i])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='r')
    plt.show()


def run_spectral():
    X = get_data()
    num_clusters = 6
    sigma = 5
    clusters, _ = spectral(X, num_clusters, sigma, mnn)
    clusters_pts = [X[np.argwhere(clusters == i)[:, 0]] for i in range(num_clusters)]
    for i in range(len(clusters_pts)):
        pts = clusters_pts[i]
        if len(pts):
            plt.scatter(pts[:, 0], pts[:, 1], color=color_dict[i])
    plt.show()


def main():
    run_spectral()


if __name__ == '__main__':
    main()
