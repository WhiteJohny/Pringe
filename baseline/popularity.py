class PopularRecommender:
    def __init__(self, ratings, movies):
        self.popular = (
            ratings
            .groupby("movie_id")["rating"]
            .count()
            .sort_values(ascending=False)
        )
        self.movies = movies.set_index("movie_id")

    def recommend(self, top_k=10):
        top_movies = self.popular.head(top_k).index
        return self.movies.loc[top_movies]
