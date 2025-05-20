import pygame
import AriaSketch
import torchvision
import torch
from torchvision import transforms
import random

class Digit:
    def __init__(self, image_tensor: torch.Tensor, label: int):
        self.label = label  # The correct digit (expected output)
        self.tensor = image_tensor  # Normalized tensor for NN input
        self.graphics = self._create_graphics(image_tensor)

    def _create_graphics(self, image_tensor):
        graphics = AriaSketch.Graphics(28, 28)
        graphics.load_pixels()

        # Convert normalized grayscale tensor to ARGB hex (white to black)
        for y in range(28):
            for x in range(28):
                pixel_val = image_tensor[0][y][x].item()  # Tensor is [1, 28, 28]
                gray = int((1.0 - pixel_val) * 255)  # Reverse norm if 1 is black
                argb = (0xFF << 24) | (gray << 16) | (gray << 8) | gray
                graphics.pixels[y][x] = argb

        graphics.update_pixels()
        return graphics

class Set:
    def __init__(self):
        self.digits = []

    def add(self, digit: Digit):
        self.digits.append(digit)

    def join(self, other):
        if isinstance(other, Set):
            self.digits.extend(other.digits)
        else:
            raise TypeError("Can only join with another Set")

    def __add__(self, other):
        if isinstance(other, Digit):
            new_set = Set()
            new_set.digits = self.digits + [other]
            return new_set
        elif isinstance(other, Set):
            new_set = Set()
            new_set.digits = self.digits + other.digits
            return new_set
        else:
            raise TypeError("Unsupported operand type for +")

    def remove(self, digit: Digit):
        self.digits.remove(digit)

    def __sub__(self, digit: Digit):
        new_set = Set()
        new_set.digits = [d for d in self.digits if d != digit]
        return new_set

    def split(self, batch_size: int):
        return [self.digits[i:i + batch_size] for i in range(0, len(self.digits), batch_size)]

    def __truediv__(self, batch_size: int):
        split_sets = self.split(batch_size)
        return [Set.from_list(batch) for batch in split_sets]

    def __len__(self):
        return len(self.digits)

    def __iter__(self):
        return iter(self.digits)

    def __getitem__(self, index: int):
        return self.digits[index]

    def to_torch_format(self):
        inputs = torch.stack([d.tensor for d in self.digits])
        labels = torch.tensor([d.label for d in self.digits], dtype=torch.long)
        return inputs, labels
    
    def shuffle(self):
        random.shuffle(self.digits)
    
    @staticmethod
    def from_list(digit_list):
        new_set = Set()
        new_set.digits = digit_list
        return new_set

def load(split_train_and_test: bool = False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_set = Set()
    for img, label in train_dataset:
        train_set.add(Digit(img, label))

    if split_train_and_test:
        test_set = Set()
        for img, label in test_dataset:
            test_set.add(Digit(img, label))
        return train_set, test_set
    else:
        train_set.join(Set.from_list([Digit(img, label) for img, label in test_dataset]))
        return train_set
