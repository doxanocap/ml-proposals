import numpy as np
import os
import json
from . import perceptron
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import torch


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


class Network:
    def __init__(self, sizes=None, model_type="native"):
        if sizes is None:
            sizes = [784, 30, 10]
        self.model_type = model_type
        self.image_id = 0

        if self.model_type == "pytorch":
            model = perceptron.load_model("./data/torch_model.txt")
            self.model = model
        else:
            self.sizes = sizes
            self.storage = {}
            self.biases = []
            self.weights = []
            self.storage_path = "./data/storage.json"
            self.load_params()

    def evaluate_img(self, image_bytes):
        image = Image.open(BytesIO(image_bytes))
        image = image.convert('L')

        self.image_id += 1
        resized_image = image.resize((28, 28))
        resized_image.save("./data/tests/image" + str(self.image_id) + ".png", "PNG")

        rg = 10
        if self.model_type == "native":
            image_data = np.asarray(resized_image)
            a = image_data.reshape(-1, 1)
            for w, b in zip(self.weights, self.biases):
                a = sigmoid(np.dot(w, a) + b)
            percentages_list = [v[0] * 100 / float(np.sum(a)) for v in a]
            y_pred = int(np.argmax(a))
        else:
            rg = 11
            output = self.model.evaluate_img(resized_image)
            probabilities = torch.softmax(output, dim=1)
            percentages_list = (probabilities * 100).squeeze().tolist()
            _, y = output.max(1)
            y_pred = y.item()

        plt.bar(range(rg), percentages_list, color="blue")
        plt.xticks(range(0, rg))
        plt.xlabel("number")
        plt.ylabel("similarity")

        plt.savefig(f"./data/tests/plot{self.image_id}.png", format="png")
        plt.clf()
        return y_pred, self.image_id

    def save_params(self):
        current_params = {
            'sizes': self.sizes,
            'biases': [item.tolist() if isinstance(item, np.ndarray) else item for item in self.biases],
            'weights': [item.tolist() if isinstance(item, np.ndarray) else item for item in self.weights]
        }

        if os.path.exists(self.storage_path):
            file = open(self.storage_path, 'r')
            file_content = file.read()
            file.close()

            if file_content != "" and len(file_content) != 0:
                self.storage = json.loads(file_content)
                for case in self.storage:
                    if case['sizes'] == self.sizes:
                        case['biases'] = current_params['biases']
                        case['weights'] = current_params['weights']
                return

        file = open(self.storage_path, 'w')
        json.dump([current_params], file, sort_keys=True, indent=4)
        file.close()

    def load_params(self):
        if os.path.exists(self.storage_path):
            file = open(self.storage_path, 'r+')
            file_content = file.read()
            file.close()

            if file_content != "" and len(file_content) != 0:
                self.storage = json.loads(file_content)
                for case in self.storage:
                    if case['sizes'] == self.sizes:
                        self.biases = [np.array(item) for item in case['biases']]
                        self.weights = [np.array(item) for item in case['weights']]
                return

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.eval(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
