#!/usr/bin/env python3
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
from torch import nn

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def get_dataloader_workers():
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="/tmp", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="/tmp", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def test_images(imgs, labels, preds):
    _, axes = plt.subplots(1, len(imgs))
    for (img, label, pred, ax) in zip(imgs, labels, preds, axes):
        if torch.is_tensor(img):
            ax.imshow(img[0].numpy())
        else:
            ax.imshow(img[0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(label + '\n' + pred)
    plt.show()

if __name__ == "__main__":
    batch_size, lr, num_epochs = 256, 0.1, 10
    net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.Softmax(1))
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    for i in range(50):
        train_iter, _ = load_data_fashion_mnist(batch_size=batch_size)
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        print(f'epoch {i + 1}, loss {l:f}')
    net.eval()
    _, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    batch = next(iter(test_iter))
    imgs, labels = batch[0], batch[1]
    preds = net(imgs).argmax(axis=1)
    test_images(imgs[:8], get_fashion_mnist_labels(labels[:8]), get_fashion_mnist_labels(preds[:8]))