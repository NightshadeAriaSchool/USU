import Import
import AriaSketch
import MNIST
import Naming

pygame = Import.do_import("pygame")
torch = Import.do_import("torch")
torchvision = Import.do_import("torchvision")
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from MNIST import Set, Digit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_name():
    adjectives = ['Swift', 'Silent', 'Clever', 'Brave', 'Wise', 'Fuzzy', 'Lucky']
    nouns = ['Fox', 'Cat', 'Owl', 'Wolf', 'Dragon', 'Hawk', 'Panther']
    return f"{random.choice(adjectives)}{random.choice(nouns)}"

class Epoch:
    def __init__(self, model, data: Set, loss_log: list[float]):
        self.model = model
        self.data = data
        self.loss_log = loss_log

class Test:
    def __init__(self, data: Set, predictions: list[int], accuracy: float):
        self.data = data
        self.predictions = predictions
        self.accuracy = accuracy

class Classifier(nn.Module):
    def __init__(self, architecture: list[int] = None, previous=None):
        super().__init__()
        self.previous = previous
        self.epoch = None
        self.name = generate_name()
        self.architecture = architecture or [784, 128, 64, 10]  # Default for MNIST
        self._build_model()

    def _build_model(self):
        layers = []
        for i in range(len(self.architecture) - 1):
            layers.append(nn.Linear(self.architecture[i], self.architecture[i + 1]))
            if i < len(self.architecture) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(device)
        x = x.view(-1, self.architecture[0])  # Flatten input
        return self.model(x)

    def clone(self, copy_only: bool = True) -> "Classifier":
        cloned = copy.deepcopy(self)
        if copy_only:
            cloned.previous = None
        return cloned

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=device))

    def train_model(self, data: Set, batch_size: int = 0) -> "Classifier":
        self.to(device)
        self.train()

        inputs, labels = data.to_torch_format()
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        loss_log = []

        if batch_size <= 0:
            batch_size = len(data)

        batches = data / batch_size
        for batch in batches:
            x_batch, y_batch = batch.to_torch_format()
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = self(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())

        self.epoch = Epoch(self.clone(copy_only=False), data, loss_log)
        return self

    def test(self, data: Set | Digit) -> Test:
        self.eval()
        with torch.no_grad():
            if isinstance(data, Digit):
                x = data.tensor.unsqueeze(0).to(device)
                output = self(x)
                pred = torch.argmax(output, dim=1).item()
                acc = float(pred == data.label)
                return Test(Set.from_list([data]), [pred], acc)
            elif isinstance(data, Set):
                inputs, labels = data.to_torch_format()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct = (preds == labels).sum().item()
                accuracy = correct / len(data)
                return Test(data, preds.tolist(), accuracy)

class DigitClassifier(Classifier):
    def __init__(self, previous=None):
        # Defaults to a 784-128-64-10 MLP
        super().__init__(architecture=[784, 128, 64, 10], previous=previous)

# Test the Classifier on MNIST dataset
def test_classifier():
    # Load MNIST dataset
    mnist = MNIST.load()
    torch_inputs, torch_labels = mnist.to_torch_format()
    