from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class ItemBasedCF:
    def __init__(self, ratings):
        self.ratings = ratings

        pivot = ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
        self.pivot = pivot
        self.matrix = csr_matrix(pivot.values)

        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.matrix.T)

    def recommend(self, movie_id, top_k=10):
        movie_idx = list(self.pivot.columns).index(movie_id)
        distances, indices = self.model.kneighbors(self.matrix.T[movie_idx], n_neighbors=top_k+1)

        similar_ids = [self.pivot.columns[i] for i in indices.flatten()[1:]]
        return similar_ids
