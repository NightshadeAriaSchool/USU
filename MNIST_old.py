from __future__ import annotations

def install_package(package: str):
    """Install a package."""
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import torch
    import torchvision
except ImportError:
    install_package("torch")
    install_package("torchvision")
    import torchvision
    import torch
import torchvision.datasets as datasets
import numpy as np

class Digit:
    def __init__(self, label: int, pixels: list[int]):
        self.label = label
        self.pixels = np.array(pixels).reshape(28, 28)
        self.nn_data = pixels
        self.nn_label = [1 if self.label == i else 0 for i in range(10)]

class Set:
    def __init__(self, digits: list[Digit]):
        self.digits = digits

    def __len__(self):
        return len(self.digits)
    
    def __getitem__(self, index: int):
        return self.digits[index]
    
    def __iter__(self):
        return iter(self.digits)
    
    def subset(self, start: int, end: int) -> Set:
        """Return a subset of the set."""
        return Set(self.digits[start:end])
    
    def subset_digit(self, digit: int) -> Set:
        """Return a subset of the set with the given digit."""
        return Set([d for d in self.digits if d.label == digit])
    
    def batches(self, batch_size: int) -> list[Set]:
        """Return a list of batches of the set."""
        batches = []
        for i in range(0, len(self.digits), batch_size):
            batches.append(self.subset(i, i + batch_size))
        return batches
    
    def shuffle(self) -> Set:
        """Shuffle the set."""
        np.random.shuffle(self.digits.copy())
        return self
    
    def copy(self) -> Set:
        """Return a copy of the set."""
        return Set(self.digits.copy())
    
    @staticmethod
    def join(*args: Set) -> Set:
        """Join multiple sets into one."""
        digits = []
        for s in args:
            digits.extend(s.digits)
        return Set(digits)
    
    def convert_to_training_data(self):
        """Convert the set to training data."""
        return np.array([d.nn_data for d in self.digits]), np.array([d.nn_label for d in self.digits])

def load():
    """Load the MNIST dataset."""
    # Load data from the library
    train = datasets.MNIST(root=".", train=True, download=True)
    test = datasets.MNIST(root=".", train=False, download=True)

    #Convert to our format
    train_digits = [Digit(d[1], d[0]) for d in train]
    test_digits = [Digit(d[1], d[0]) for d in test]
    digits = train_digits + test_digits
    return Set(digits)

DATA = load()
print(len(DATA))