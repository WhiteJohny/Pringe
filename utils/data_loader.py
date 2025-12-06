import pandas as pd

def load_movielens_1m(path="data/"):
    ratings = pd.read_csv(
        path + "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )

    movies = pd.read_csv(
        path + "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )

    users = pd.read_csv(
        path + "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1"
    )

    return ratings, movies, users
