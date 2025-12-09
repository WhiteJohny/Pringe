import time
import torch
from torch import nn
from two_tower import device, process_dataset, TwoTower, TwoTowerRecommender
from utils.data_loader import load_movielens_1m


def train_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = (X[0].to(device), X[1].to(device)), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.float())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X[0])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d} ({current/size*100:.2f}%)]")


def test_model(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = (X[0].to(device), X[1].to(device)), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.float()).item()
    test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")


def fit(model, loss_fn, optimizer, train_dataloader, test_dataloader, epochs=5):
    for t in range(epochs):
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {t+1}\n-------------------------------")
        train_model(train_dataloader, model, loss_fn, optimizer)
        test_model(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == '__main__':
    load_model = input("Try to load existing model? (y/n):")

    ratings, movies, users = load_movielens_1m()
    train_dataloader, test_dataloader = process_dataset(ratings, movies, users)

    if load_model.lower() == "y":
        model = torch.load("data/two_tower_model.pth", weights_only=False).to(device)
    else:
        model = TwoTower().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
    fit(model, loss_fn, optimizer, train_dataloader, test_dataloader, epochs=100)

    torch.save(model, "data/two_tower_model.pth")

    two_tower = TwoTowerRecommender(users, movies, model)
    two_tower_df = two_tower.recommend(user_id=10, top_k=5)
    print("\nTWO-TOWER RECOMMENDER\n")
    print(two_tower_df.to_string(index=False))
