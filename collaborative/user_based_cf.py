from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class UserBasedCF:
    def __init__(self, ratings):
        pivot = ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
        self.pivot = pivot
        self.matrix = csr_matrix(pivot.values)

        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.matrix)

    def find_similar_users(self, user_id, top_k=5):
        user_idx = list(self.pivot.index).index(user_id)
        distances, indices = self.model.kneighbors(self.matrix[user_idx], n_neighbors=top_k+1)
        return indices.flatten()[1:]

    def recommend(self, user_id, top_k=10):
        similar_users = self.find_similar_users(user_id)

        user_data = self.pivot.loc[similar_users].mean(axis=0)
        user_data_sorted = user_data.sort_values(ascending=False)

        return user_data_sorted.head(top_k).index.to_list()
