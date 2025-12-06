from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, movies):
        self.movies = movies.reset_index()

        self.vectorizer = TfidfVectorizer(token_pattern='[A-Za-z]+')
        self.tfidf = self.vectorizer.fit_transform(movies["genres"])

        self.sim_matrix = cosine_similarity(self.tfidf)

    def recommend(self, movie_id, top_k=10):
        idx = self.movies[self.movies.movie_id == movie_id].index[0]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:top_k+1]
        movie_indices = [i[0] for i in sim_scores]

        return self.movies.iloc[movie_indices][["movie_id", "title", "genres"]]
