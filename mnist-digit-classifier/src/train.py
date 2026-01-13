from src.mnist import pickle
from src.models import network, sgd

if __name__ == '__main__':
    training_data, validation_data, test_data = pickle.load()

    network = network.Network([784, 64, 10])
    model = sgd.SGD(network, training_data, test_data=test_data,
                    eta=3, epochs=30, batch_size=10)
    model.run_training()
