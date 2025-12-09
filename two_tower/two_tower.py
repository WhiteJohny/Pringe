import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


CONFIG = {
    'user_embedding_num': 6041,
    'user_embedding_dim': 100,
    'item_embedding_num': 3884,
    'item_embedding_dim': 100,
    'user_dense': [50, 60, 20],
    'item_dense': [50, 60, 20]
}

user_features = [
    "user_id", "age", "gender", "occupation", "zip"
]
item_features = [
    "movie_id", "Action", "Romance", "Thriller", "War", "Horror", "Musical", "Western", "Comedy",
    "Documentary", "Children's", "Fantasy", "Adventure", "Crime", "Mystery", "Sci-Fi", "Animation",
    "Film-Noir", "Drama"
]


class TwoTower(nn.Module):
    def __init__(self):
        super(TwoTower, self).__init__()
        self.user_embedding_num = CONFIG["user_embedding_num"]
        self.user_embedding_dim = CONFIG["user_embedding_dim"]

        self.item_embedding_num = CONFIG["item_embedding_num"]
        self.item_embedding_dim = CONFIG["item_embedding_dim"]

        self.user_dense = [len(user_features) * self.user_embedding_dim, *CONFIG["user_dense"]]
        self.item_dense = [len(item_features) * self.item_embedding_dim, *CONFIG["item_dense"]]
        user_dense_layers = []
        item_dense_layers = []
        self.flatten = nn.Flatten()
        self.user_embedding = nn.Embedding(self.user_embedding_num, self.user_embedding_dim)
        for i in range(len(self.user_dense) - 1):
            dense = nn.Linear(self.user_dense[i], self.user_dense[i+1])
            act = nn.ReLU()

            user_dense_layers.append(dense)
            user_dense_layers.append(act)

        self.user_tower = nn.Sequential(*user_dense_layers)

        self.item_embedding = nn.Embedding(self.item_embedding_num, self.item_embedding_dim)
        for i in range(len(self.item_dense) - 1):
            dense = nn.Linear(self.item_dense[i], self.item_dense[i+1])
            act = nn.ReLU()

            item_dense_layers.append(dense)
            item_dense_layers.append(act)

        self.item_tower = nn.Sequential(*item_dense_layers)

    def forward(self, X):
        user_embed = self.user_embedding(X[0])
        user_embed = self.flatten(user_embed)
        item_embed = self.item_embedding(X[1])
        item_embed = self.flatten(item_embed)

        user = self.user_tower(user_embed)
        item = self.item_tower(item_embed)
        score = torch.dot(user.reshape((-1,)), item.reshape((-1,)))
        return score
