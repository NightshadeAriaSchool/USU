from __future__ import annotations
import Import

pygame = Import.do_import("pygame")
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import Naming  # for naming NNs or reports, if needed
import MNIST
from MNIST import Set, Digit

class TrainingReport:
    def __init__(self, dataset: Set, loss_value: float, learning_rate=float):
        self.dataset = dataset
        self.loss = loss_value
        self.learning_rate = learning_rate

    def __str__(self):
        return f"Trained on {len(self.dataset)} samples, final loss: {self.loss:.4f}"

class TestingReport:
    def __init__(self, dataset: Set, outputs: torch.Tensor):
        self.dataset = dataset
        self.expected_labels = torch.tensor([d.label for d in dataset])
        self.predicted_labels = outputs
        self.expected_digits = torch.tensor([d.digit for d in dataset])
        self.predicted_digits = torch.argmax(outputs, dim=1)
        self.correct = (self.predicted_digits == self.expected_digits).sum().item()
        self.total = len(dataset)
        self.accuracy = self.correct / self.total

    def __str__(self):
        return f"Tested on {self.total} samples, accuracy: {self.accuracy:.2%}"

class Classifier:
    def __init__(self, layer_sizes=None, prev=None, report=None, model=None):
        self.name = Naming.new_name()
        if layer_sizes is None:
            self.name = prev.name.increment() if not prev.trained else prev.name.fork()
            self.layer_sizes = prev.layer_sizes
            self.prev = prev
            self.prev.trained = True
            self.report = report
            self.model = model
        else:
            self.name = Naming.new_name()
            self.layer_sizes = layer_sizes
            self.prev = prev
            self.report = report
            self.model = self._build_network([784, *layer_sizes, 10])
        self.trained = False
        
    def __str__(self):
        lines = [f"🧠 Classifier: {self.name}"]
        if self.report:
            lines.append(f"📉 Loss: {getattr(self.report, 'loss', 'N/A'):.6f}")
            lines.append(f"🎓 Learning rate: {getattr(self.report, 'learning_rate', 'N/A')}")
        else:
            lines.append("📉 Loss: <no report>")
        lines.append("🔍 Model parameters:")
        for i, param in enumerate(self.model.parameters()):
            data = param.data
            lines.append(
                f"  Layer {i:02d}: shape={tuple(data.shape)} | min={data.min().item():.4f} | "
                f"max={data.max().item():.4f} | mean={data.mean().item():.4f}"
            )
        if self.prev and self.prev == self:
            lines.append(f"Model parameters equal the previous one's.")
        else:
            lines.append(f"Model parameters differ from the last one's.")
        return "\n".join(lines)
    def __eq__(self, value):
        for p1, p2 in zip(self.model.parameters(), value.model.parameters()):
            if not torch.equal(p1, p2):
                return False
            return True
    
    @property
    def forked_by(self):
        current = self
        id = self.name.id
        while current and current.name.id == id:
            current = current.prev
        return current
    
    def get_epoch(self, epoch: int):
        if epoch > self.name.epoch:
            return self
        current = self
        while current and current.name.epoch != epoch:
            current = current.prev
        return current
    
    def _build_network(self, layer_sizes):
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
            if i == 0:
                layers.append(nn.Dropout(0.2))
                
        return nn.Sequential(*layers)
    def _build_network2(self, layer_sizes):
        return nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def predict(self, data: Set|Digit) -> torch.Tensor:
        if isinstance(data, Digit):
            data = Set([data])
        
        torch_data = data.to_torch_format()
        if isinstance(torch_data, tuple):
            torch_data = torch_data[0]  # Only use the input tensor
        with torch.no_grad():
            outputs = self.model(torch_data)
        
        return torch.softmax(outputs, dim=1)
    
    def train(self, dataset: Set, learning_rate: float = 0.01) -> Classifier:
        new_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(new_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        inputs, labels = dataset.to_torch_format()
        optimizer.zero_grad()
        outputs = new_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Create the return Classifier
        return Classifier(prev=self, report=TrainingReport(dataset, loss.item(), learning_rate), model=new_model)
    
    def test(self, dataset: Set) -> TestingReport:
        with torch.no_grad():
            outputs = self.model(dataset.to_torch_format())
        return TestingReport(dataset, outputs)
