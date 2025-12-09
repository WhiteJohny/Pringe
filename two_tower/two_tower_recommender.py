import torch
import pandas as pd

from .two_tower import item_features, device


class TwoTowerRecommender:
    def __init__(self, users: pd.DataFrame, movies: pd.DataFrame, model=None):
        users = users.copy().fillna(0)
        movies = movies.copy().fillna(0)

        genres = movies["genres"].str.get_dummies(sep="|")
        movies = pd.concat([movies.drop("genres", axis=1), genres], axis=1)

        users['gender'] = users['gender'].map(lambda x: 0 if x == 'M' else 1)
        users['zip'] = users['zip'].map(lambda x: x[0] if x[0].isdigit() else 9)
        users = users.astype('int')

        self.users: pd.DataFrame = users
        self.movies: pd.DataFrame = movies

        self.model = model or torch.load("data/two_tower_model.pth", weights_only=False).to(device)
        self.model.eval()

    def recommend(self, user_id, top_k=10):
        with torch.no_grad():
            user = (torch.Tensor(self.users.loc[self.users["user_id"] == user_id].values.flatten().tolist())
                    .to(device=device, dtype=torch.int))

            items = torch.Tensor(self.movies[item_features].to_numpy()).to(device, dtype=torch.int)
            users = user.repeat(items.size(0), 1)

            X = (users, items)
            y = self.model(X)

            y = y.cpu().numpy()

            self.movies["score"] = y
            results = self.movies.nlargest(top_k, columns="score")

            return results[["movie_id", "title", "score"]]
