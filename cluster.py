import numpy as np

RANDOM_SEED = 12345678
np.random.seed(RANDOM_SEED)

class cluster:

    def __init__(self, k=5, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations


    def fit(self, X):
        clusters = [None] * len(X)
        centroids = []
        # find out the range of the dataset
        max_elem = np.max(X)
        min_elem = np.min(X)

        # generate random k centroids
        centroids = np.random.uniform(min_elem, max_elem, (self.k, len(X[0]))).tolist()

        for _ in range(self.max_iterations):
            
            # Assignment 
            for i in range(len(X)):
                min_dist = float('inf')
                closest_centroid_idx = None
                for j in range(self.k):
                    p1 = np.array(X[i])
                    p2 = np.array(centroids[j])
                    dist = np.linalg.norm(p1 - p2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_centroid_idx = j
                clusters[i] = closest_centroid_idx

            # Update
            new_centroids = centroids
            for i in range(self.k):
                cluster = [X[j] for j in range(len(X)) if clusters[j] == i]
                new_centroids[i] = np.mean(cluster, axis=0).tolist()
            
            # checking for convergence
            # if np.array_equal(centroids, new_centroids):
            #     break
            # else:
            #     centroids = new_centroids

        return clusters, centroids

# cluster = cluster(2)
# X = [ [0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10] ]
# clusters, centroids = cluster.fit(X)
# print(clusters)
# print(centroids)