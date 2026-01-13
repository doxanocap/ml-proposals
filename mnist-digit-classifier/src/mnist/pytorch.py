from torchvision import datasets
from torchvision import transforms
import random


def pytorch():
    MNIST_train = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
    MNIST_test = datasets.MNIST('./mnist', train=False, download=True, transform=transforms.ToTensor())

    FMNIST_train = datasets.FashionMNIST(root='./fmnist', train=True, download=True, transform=transforms.ToTensor())
    FMNIST_test = datasets.FashionMNIST(root='./fmnist', train=False, download=True, transform=transforms.ToTensor())

    CMNIST_train = [[FMNIST_train[idx][0], 10] for idx in range(6000)]
    CMNIST_train.extend(MNIST_train)
    random.shuffle(CMNIST_train)

    CMNIST_test = [[FMNIST_test[idx][0], 10] for idx in range(1000)]
    CMNIST_test.extend(MNIST_test)
    random.shuffle(CMNIST_test)
    return CMNIST_train, CMNIST_test
