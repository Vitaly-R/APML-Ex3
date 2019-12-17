import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from sklearn.decomposition import PCA
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# A color dictionary used when visualizing clustering results.
not_include = ['whitesmoke', 'white', 'snow', 'mistyrose', 'seashell', 'peachpuff', 'linen', 'bisque', 'red', 'antiquewhite', 'oldlace', 'floralwhite', 'cornsilk', 'lemonchiffon', 'ivory',
               'beige', 'lightyellow', 'lightgoldenrodyellow', 'lawngreen', 'honeydew', 'palegreen', 'mintcream', 'azure', 'lightcyan', 'paleturquoise', 'powderblue', 'skyblue', 'aliceblue',
               'lightslategrey', 'lightslategray', 'slategrey', 'slategray', 'ghostwhite', 'lavender', 'dimgray', 'dimgrey', 'darkgrey', 'darkgray', 'lightgray', 'gainsboro', 'darkmagenta',
               'lavenderblush', 'pink']
color_dict = {i: c for (i, c) in enumerate([color for color in colors.CSS4_COLORS if color not in not_include])}


def circles_example():
    """
    An example function for generating and plotting synthetic data.
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
    An example function for loading and plotting the APML_pic data.
    :param path: the path to the APML_pic.pickle file
    """
    with open(path, 'rb') as f:
        apml = pickle.load(f)
    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle', genes_path='microarray_genes.pickle', conds_path='microarray_conds.pickle'):
    """
    An example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the function.
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
    Calculating the pair-wise euclidean distance between two groups of vectors in the same dimansion.
    :param X: A NxD matrix.
    :param Y: A MxD matrix.
    :return: NxM euclidean distance matrix representing the pair-wise euclidean distance between two data matrices.
    """
    return distance.cdist(X, Y, 'euclidean')


def euclidean_centroid(X):
    """
    Calculate the centroid of the cluster X.
    :param X: A sub-matrix of the NxD data matrix that defines a cluster.
    :return: The vector closest to the true center of mass of X.
    """
    center_of_mass = np.average(X, axis=0)
    return X[np.argmin(np.linalg.norm(X - center_of_mass, axis=1))]


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of k-means++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: A metric function like specified in the k-means documentation.
    :return: A kxD matrix with rows containing the centroids.
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
    An implementation of the K-Means algorithm, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their pair-wise distance.
    :param center: A function that accepts a sub-matrix of X where the rows are points in a cluster,
                    and returns the clusters centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :return: A tuple of (clustering, centroids) -
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
    Calculating the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """
    return np.exp(-(X ** 2) / (2 * (sigma ** 2)))


def mnn(X, m):
    """
    Calculating the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    nearest_neighbors_inds = X.argsort(axis=1)[:, 1: (m + 1)]  # this includes the point X[i] itself, thus for the m nearest neighbors, we use the indices in places {1,...,m}
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
    :return: Clustering, as in the kmeans implementation.
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
    normalized_k_eigenvectors = k_eigenvectors / np.linalg.norm(k_eigenvectors, axis=1).reshape((k_eigenvectors.shape[0], 1))  # projecting rows onto the unit sphere
    normalized_k_eigenvectors = np.nan_to_num(normalized_k_eigenvectors)
    clusters, centroids = kmeans(normalized_k_eigenvectors, k, 20)
    inds = list()
    for v in centroids:
        inds.append(np.unique(np.argwhere(normalized_k_eigenvectors == v)[:, 0])[0])
    return clusters, X[inds]


def cost(X, clusters, centroids):
    """
    Calculates the cost for the given clustering of data X around the given centroids.
    :param X: Data points, a numpy array of shape (N, d)
    :param clusters: Clustering array of shape (N, )
    :param centroids: The centroids corresponding to the clusters (meaning, centroids[i] is the centroid of the points in the array X[np.argwhere(clusters == i)[:, 0]]
    :return: The cost as the sum of distances between the points in each cluster.
    """
    total = 0
    for i in range(centroids.shape[0]):
        inds = np.argwhere(clusters == i)
        inds = inds[:, 0] if len(inds.shape) == 2 else inds
        cluster = X[inds]
        total += np.sum(euclid(cluster, cluster))
    return total


def generate_circles():
    """
    Generates a synthetic dataset of 4 concentric circles.
    :return: An array of shape (N, 2) of the points in the dataset.
    """
    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.array([np.cos(t) + 0.1 * np.random.randn(length), np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.array([2 * np.cos(t) + 0.1 * np.random.randn(length), 2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.array([3 * np.cos(t) + 0.1 * np.random.randn(length), 3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.array([4 * np.cos(t) + 0.1 * np.random.randn(length), 4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    return circles.T


def generate_apml_pic():
    """
    Load the APML pic dataset.
    :return: An array of shape (N, 2) of the points in the dataset.
    """
    with open('APML_pic.pickle', 'rb') as f:
        apml = pickle.load(f)
    return apml


def generate_microarray():
    """
    Load the microarray dataset.
    :return: An array of shape (N, d) of the points in the dataset.
    """
    with open('microarray_data.pickle', 'rb') as f:
        data = pickle.load(f)
    return data


def get_data(circles=True, apml=True):
    """
    Loads and returns a dataset.
    :param circles: Weather the dataset should be the circles dataset.
    :param apml: Weather the dataset should be the APML pic dataset.
    :return: The requested dataset if circles or apml are True, otherwise, the microarray dataset.
    """
    if circles:
        return generate_circles()
    elif apml:
        return generate_apml_pic()
    else:
        return generate_microarray()


def plot_clusters(X, clusters, num_clusters, title=''):
    """
    Plots in 2D the clustering of X into the given clusters, with different colors for each cluster.
    :param X: A dataset of 2D points (an array of shape (N, 2))
    :param clusters: An array of shape (N, ) representing the clusters of the corresponding poitns in X.
    :param num_clusters: The number of clusters.
    :param title: A title for the plot.
    """
    clusters_pts = [X[np.argwhere(clusters == i)[:, 0]] for i in range(num_clusters)]
    plt.figure()
    plt.title(title)
    for i in range(len(clusters_pts)):
        pts = clusters_pts[i]
        if len(pts):
            plt.scatter(pts[:, 0], pts[:, 1], color=color_dict[i])


def run_clustering(X, num_clusters=4, sigma=5, similarity=gaussian_kernel, title='', as_spectral=True):
    """
    Runs a clustering of the given X with the given parameters, and plots the resulting clusters in 2D.
    :param X: The dataset of points to cluster.
    :param num_clusters: The number of resulting clusters desired.
    :param sigma: The similarity parameter for the similarity function.
    :param similarity: The similarity function.
    :param title: A title for the plot of the clusters
    :param as_spectral: Weather to use spectral clustering or regular k-means.
    """
    clusters, _ = spectral(X, num_clusters, sigma, similarity) if as_spectral else kmeans(X, num_clusters)
    plot_clusters(X, clusters, num_clusters, title)


def plot_vs_k(X, func=cost, max_k=100, title='', ylabel=''):
    """
    Plots the values of the cost function as a function of k.
    The cost is calculated according to the results of clustering with the k-means algorithm, and the given cost function.
    The values of k are from 1 to the max_k (including) in steps of 5.
    :param X: The data to cluster.
    :param func: The cost function with which to calculate.
    :param max_k: The max value of k for which we calculate the cost.
    :param title: Title of the plot.
    :param ylabel: Label for the y axis (since we would also like the option to calculate other parameters as a function of k).
    """
    ks = np.arange(0, max_k + 1, 5)
    ks[0] = 1
    results = list()
    for k in ks:
        clusters, centroids = kmeans(X, k)
        results.append(func(X, clusters, centroids))
    plt.figure()
    plt.title(title)
    plt.xlabel('k')
    plt.ylabel(ylabel)
    plt.plot(ks, results)


def plot_clusters_3d(X, clusters, k, title=''):
    """
    Plots the clusters of the given dataset in 3D space.
    This function is used (in the context of this exercise) with PCA or t-SNE which embed high dimensional data into 3 dimensions
    for us to visualize.
    :param X: A group of points in 3D space (An array of shape (N, 3)).
    :param clusters: Clustering of the points (An array of shape (N, )).
    :param k: The number of clusters.
    :param title: A title for the plot.
    """
    clusters_pts = [X[np.argwhere(clusters == i)[:, 0]] for i in range(k)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for i in range(k):
        pts = clusters_pts[i]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color_dict[i])


def create_high_d_synthetic_data(d=100, groups=10):
    """
    This function generates a synthetic high dimensional data set.
    The defined structure on the dataset is groups of points (the number of groups is passed as a parameter) around random centers.
    :param d: The requested dimension.
    :param groups: The number of groups to generate
    :return: the generated data set.
    """
    group_size = 100
    centers = np.random.multivariate_normal(np.zeros(d), np.eye(d), groups)
    data = np.zeros((groups * group_size, d))
    for i in range(groups):
        group = np.random.multivariate_normal(centers[i], np.eye(d), group_size)
        data[i * group_size: (i + 1) * group_size] = group
    np.random.shuffle(data)
    return data


def q_2_4():
    """
    Uses spectral clustering to cluster synthetic data using different values of sigma (the similarity parameter),
    and different similarity transformations (symmetric MNN, and heat kernel).
    In addition, clusters each dataset using regular k-means clustering.
    After clustering, the clusters are plotted in 2D.
    This function helps us visualize the effect of different values of sigma on the resulting clusters for different similarity measures.
    """
    sigmas = [1, 5, 10, 15, 20]
    # Clustering the circles dataset into 4 clusters.
    X = get_data()
    run_clustering(X, 4, title='K-Means Clustering', as_spectral=False)
    for sigma in sigmas:
        run_clustering(X, sigma=sigma, title='Spectral Clustering \nGaussian Kernel \nsigma=' + str(sigma))
        run_clustering(X, sigma=sigma, similarity=mnn, title='Spectral Clustering \nMNN \nsigma=' + str(sigma))
    # Clustering the APML pic dataset into 9 clusters.
    X = get_data(False)
    run_clustering(X, 9, title='K-Means Clustering', as_spectral=False)
    for sigma in sigmas:
        run_clustering(X, num_clusters=9, sigma=sigma, title='Spectral Clustering \nGaussian Kernel \nsigma=' + str(sigma))
        run_clustering(X, num_clusters=9, sigma=sigma, similarity=mnn, title='Spectral Clustering \nMNN \nsigma=' + str(sigma))


def q_2_4_1():
    """
    Plots of the similarity graph on the circles data set on shuffled data, and on data which was sorted according to the given clusters.
    This function helps us visualize the information uncovered by the spectral clustering about the connection between points in the data.
    """
    sigma = 10
    # Circles data with k = 4
    X = get_data()
    np.random.shuffle(X)
    shuffled_data_distance_matrix = euclid(X, X)
    shuffled_data_similarity_matrix = gaussian_kernel(shuffled_data_distance_matrix, sigma)
    clusters, _ = spectral(X, k=4, similarity_param=sigma)
    X_by_clusters = X[np.argsort(clusters)]
    clustered_data_distance_matrix = euclid(X_by_clusters, X_by_clusters)
    clustered_data_similarity_matrix = gaussian_kernel(clustered_data_distance_matrix, sigma)
    plt.figure()
    plt.title('Similarity Matrix\nShuffled Data')
    plt.imshow(shuffled_data_similarity_matrix, cmap='hot', extent=[0, len(X), 0, len(X)])
    plt.colorbar()
    plt.figure()
    plt.title('Similarity Matrix\nData Ordered by Clusters')
    plt.imshow(clustered_data_similarity_matrix, cmap='hot', extent=[0, len(X), 0, len(X)])
    plt.colorbar()


def q_2_5_1():
    """
    This function runs k-means clustering on different values of k, and plots the resulting cost as a function of k.
    The calculation is done over the APML pic dataset.
    This function demonstrates the elbow method to choose k - since the cost will drop sharply at first, and for higher
    values of k, the decrease in cost will become more gradual, we would like to choose k which is at that transition (at the elbow).
    """
    X = get_data(False)
    plot_vs_k(X, title='Clustering Cost\nAPML Pic Dataset', ylabel='cost')


def q_2_6():
    """
    Preforming spectral clustering and k-means clustering on the microarray data set.
    The value of k is chosen using the elbow method (visualized in the commented line).
    The value of sigma is chosen from the values used in the function q_4, in which we
    saw the effect of different values of sigma on the resulting clustering.
    Finally, a visualization of the resulting clustering is done using PCA from sklearn into 3 dimensions.
    """
    X = get_data(False, False)
    # plot_vs_k(X, title='Cost of clustering vs k on micro array data', ylabel='cost')  # Uncomment to see the effect of k on the cost of the clustering.
    k = 15
    sigma = 5
    km_clusters, km_centroids = kmeans(X, k)
    sp_clusters, sp_centroids = spectral(X, k, sigma, mnn)
    pca = PCA(n_components=3)
    transformed_X = pca.fit_transform(X, km_clusters)
    plot_clusters_3d(transformed_X, km_clusters, k, 'PCA to 3 dimensions of k-means clustering of microarray dataset')
    plot_clusters_3d(transformed_X, sp_clusters, k, 'PCA to 3 dimensions of spectral clustering of microarray dataset')


def q_2_7():
    """
    Preforms spectral clustering on a synthetic high-dimensional dataset, and uses TSNE (from sklearn)
    to embed the resulting clusters into 3 dimensions and visualizes the result in 3D.
    """
    k = 10
    sigma = 5
    X = create_high_d_synthetic_data(groups=k)
    km_clusters, _ = spectral(X, k, sigma, mnn)
    tsne = TSNE(n_components=3)
    embedded = tsne.fit_transform(X, km_clusters)
    plot_clusters_3d(embedded, km_clusters, 10, 'TSNE to 3 dimensions of spectral clustering of synthetic high dimensional dataset')


def main():
    q_2_4()
    q_2_4_1()
    q_2_5_1()
    q_2_6()
    q_2_7()
    plt.show()


if __name__ == '__main__':
    main()
