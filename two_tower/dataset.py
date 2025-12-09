import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user_features: list, item_features: list, label: list):
        self.user_features = df[user_features].values
        self.item_features = df[item_features].values
        self.rating = df[label].values

    def __len__(self):
        return len(self.user_features)

    def __getitem__(self, idx):

        user_features = self.user_features[idx]
        item_features = self.item_features[idx]
        rating = self.rating[idx]

        return [torch.tensor(user_features), torch.tensor(item_features)], torch.tensor(rating)


def process_dataset(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame):
    users = users.copy().fillna(0)
    movies = movies.copy().fillna(0)
    ratings = ratings.copy().fillna(0)

    genres = movies["genres"].str.get_dummies(sep="|")
    movies = pd.concat([movies.drop("genres", axis=1), genres], axis=1)

    users['gender'] = users['gender'].map(lambda x: 0 if x == 'M' else 1)
    users['zip'] = users['zip'].map(lambda x: x[0] if x[0].isdigit() else 9)
    users = users.astype('int')

    df = ratings.merge(users, left_on="user_id", right_index=True)
    df = df.merge(movies, left_on="movie_id", right_index=True)

    train_data, test_data = train_test_split(df, test_size=0.2)

    user_features = ["user_id", "age", "gender", "occupation", "zip"]
    item_features = ["movie_id"] + list(genres.columns)
    label = ["rating"]

    train_dataloader = DataLoader(CustomDataset(
        train_data,
        user_features,
        item_features,
        label), batch_size=4096, shuffle=True)

    test_dataloader = DataLoader(CustomDataset(
        test_data,
        user_features,
        item_features,
        label), batch_size=4096, shuffle=True)

    return train_dataloader, test_dataloader
