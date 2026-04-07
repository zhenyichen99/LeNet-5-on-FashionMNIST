import torch
from torch import nn
import matplotlib.pyplot as plt

def train(dataloader, model, loss_fn, optimizer, device):

    size = len(dataloader.dataset)
    model.train()
    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):

        # propagate forward
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item()

        # backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # return avg train loss per sample
    train_loss /= size
    return train_loss


def test(dataloader, model, loss_fn, device):

    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    # testing
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()

    # return avg test loss per sample and accuracy
    test_loss /= size
    correct /= size
    return test_loss, correct*100


def early_stop(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, patience, show=False):

    params = model.state_dict()
    min_loss = float('inf')
    train_losses, test_losses, accuracies = [], [], []
    n, e = 0, 0

    # print column names
    if show:
            print(f'epoch | train_loss | test_loss | accuracy(%)')

    while n < patience:

        # train and test current epoch
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        test_loss, accuracy = test(test_dataloader, model, loss_fn, device)
        if show:
            print(f'{e+1:<6}|{train_loss:>9.4f}   |{test_loss:>9.4f}  |{accuracy:>9.1f}')
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        e += 1

        # update best epoch
        if test_loss < min_loss:
            params = model.state_dict()
            min_loss = test_loss
            final_accuracy = accuracy
            n = 0
        else:
            n += 1

    # report best epoch and corresponding accuracy
    best_epoch = e-patience
    print(f'Epoch {best_epoch} has minimum test loss, with test accuracy {(final_accuracy):>.1f}%.')
    torch.save(params, 'LeNet5.pth')

    # plot train/test loss and test accuracy
    epochs = list(range(1,e+1))

    plt.subplot(2,1,1)
    plt.plot(epochs, train_losses, 'b', label='train loss')
    plt.plot(epochs, test_losses, 'g', label='test loss')
    plt.plot(best_epoch, min_loss, 'g*')
    plt.ylabel('loss')
    plt.legend()
    plt.xticks(epochs)

    plt.subplot(2,1,2)
    plt.plot(epochs, accuracies, 'r', label='test accuracy')
    plt.plot(best_epoch, final_accuracy, 'r*')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend()
    plt.xticks(epochs)

    plt.show()
    plt.close()


        





