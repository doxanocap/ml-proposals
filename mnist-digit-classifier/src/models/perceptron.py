from tqdm.notebook import tqdm
import torch
from torchvision import transforms


def load_model(path):
    model = Perceptron(num_layers=2, output_dim=11, p=0.3, hidden_dim=128)
    model.load_state_dict(torch.load(path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


class Perceptron(torch.nn.Module):
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    def __init__(self, input_dim=784, num_layers=0,
                 hidden_dim=64, output_dim=10, p=0.0):
        super(Perceptron, self).__init__()

        self.layers = torch.nn.Sequential()

        prev_size = input_dim
        for i in range(num_layers):
            self.layers.add_module('layer{}'.format(i),
                                   torch.nn.Linear(prev_size, hidden_dim))
            self.layers.add_module('relu{}'.format(i), torch.nn.ReLU())
            self.layers.add_module('dropout{}'.format(i), torch.nn.Dropout(p=p))
            prev_size = hidden_dim

        self.layers.add_module('classifier',
                               torch.nn.Linear(prev_size, output_dim))

    def evaluate_img(self, img):
        toTensor = transforms.ToTensor()
        tensor_img = toTensor(img)
        x = tensor_img.view(1, -1)
        output = self.layers(x)
        return output

    def forward(self, y):
        return self.layers(y)

    def run_training(self, dataset, loss_function=None, optimizer=None, lr=0.001, momentum=0.9, epochs=10):
        if loss_function is None:
            loss_function = torch.nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs), leave=False):
            generator = torch.utils.data.DataLoader(dataset, batch_size=64,
                                                    shuffle=True)
            for x, y in tqdm(generator, leave=False):
                optimizer.zero_grad()
                x = x.view([-1, 784]).to(self.device())
                y = y.to(self.device())

                output = self(x)
                loss = loss_function(output, y)
                loss.backward()
                optimizer.step()

        torch.save(self.state_dict(), './torch_model.txt')

    def run_testing(self, dataset):
        generator = torch.utils.data.DataLoader(dataset, batch_size=64)

        pred = []
        real = []
        for x, y in generator:
            x = x.view([-1, 784]).to(self.device())
            y = y.to(self.device())

            pred.extend(torch.argmax(self(x), dim=-1).cpu().numpy().tolist())
            real.extend(y.cpu().numpy().tolist())
