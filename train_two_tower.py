import torch
from torch import nn
from two_tower import device, process_dataset, TwoTower
from utils.data_loader import load_movielens_1m


def train_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print(size)
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

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
        print(f"Epoch {t+1}\n-------------------------------")
        train_model(train_dataloader, model, loss_fn, optimizer)
        test_model(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == '__main__':
    ratings, movies, users = load_movielens_1m()
    train_dataloader, test_dataloader = process_dataset(ratings, movies, users)

    model = TwoTower().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    fit(model, loss_fn, optimizer, train_dataloader, test_dataloader)

    torch.save(model, "data/two_tower_model.pth")

