from __future__ import annotations
import Import
pygame = Import.do_import("pygame")
torchvision = Import.do_import("torchvision")
torch = Import.do_import("torch")
import torch
from torchvision import transforms
import torch.nn.functional as F
import random
from Yui import Graphics
import numpy as np

_loading = False
_set = None
progress = 0
total = 1

class Digit:
    def __init__(self, image_tensor: torch.Tensor, digit: int = None):
        self.digit = digit
        if digit is not None:
            self.label = F.one_hot(torch.tensor(digit).long(), num_classes=10).float()
        else:
            self.label = torch.zeros(10)
        self.tensor = image_tensor.flatten()  # Normalized tensor for NN input
        self._graphics = None
    
    @property
    def graphics(self):
        if self._graphics is None:
            self._graphics = self._create_graphics(self.tensor)
        return self._graphics
    
    @staticmethod
    def from_graphics(graphics) -> 'Digit':
        # Resize to 28x28 if needed
        if graphics.get_width() != 28 or graphics.get_height() != 28:
            graphics = pygame.transform.smoothscale(graphics, (28, 28))
        # Convert Yui.Graphics to a numpy array of grayscale pixel values
        arr = np.zeros((28, 28), dtype=np.float32)
        for y in range(28):
            for x in range(28):
                g = graphics.get_at((x, y)).g
                arr[y, x] = g / 255.0
        print(arr)
        tensor = torch.from_numpy(arr).view(1, -1)  # Shape (1, 784)
        digit = Digit(tensor)
        digit._graphics = graphics
        return digit

    def _create_graphics(self, image_tensor):
        raise NotImplementedError()

class Set:
    def __init__(self, digits: list[Digit] = []):
        self.digits = digits

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

    def __getitem__(self, index: int|slice):
        if isinstance(index, slice):
            return Set(self.digits[index])
        return self.digits[index]

    def to_torch_format(self):
        inputs = torch.stack([torch.as_tensor(d.tensor) for d in self.digits])
        labels = torch.stack([torch.as_tensor(d.label) for d in self.digits])  # Shape: (batch_size, 10)
        return inputs, labels
    
    def shuffle(self, seed: int = None) -> Set:
        digits_copy = self.digits.copy()
        rng = random.Random(seed)
        rng.shuffle(digits_copy)
        return Set(digits_copy)
    
    def random(self, size: float = None, seed: int = None, return_test: bool = False) -> Set|tuple[Set, Set]:
        return self.shuffle()[:max(0, min(len(self), size if size else len(self)))]
    
    def get_labeled(self, digit: 1) -> Set:
        return Set([d for d in self if d.digit == digit])
    
    @staticmethod
    def from_list(digit_list):
        new_set = Set()
        new_set.digits = digit_list
        return new_set
    
    def summary(self) -> str:
        counts = [0] * 10  # for digits 0-9
        for digit in self.digits:
            if 0 <= digit.digit <= 9:
                counts[digit.digit] += 1
        return ', '.join(f"{i}: {count}" for i, count in enumerate(counts))
    
def load(split_train_and_test: bool = False, max_size: int = 70000):
    if _set:
        return _set
    elif _loading:
        while not _set:
            pass
        return _set
        
        
    print("Loading...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    print("Loading data...")
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    global total
    total = min(max_size, len(train_dataset) + len(test_dataset))

    print("Converting data...")
    train_set = Set()
    global progress
    for img, label in train_dataset:
        if progress == max_size:
            break
        train_set.add(Digit(img, label))
        progress += 1
    
    print("Finalizing...")
    if split_train_and_test:
        test_set = Set()
        for img, label in test_dataset:
            if progress == max_size:
                break
            test_set.add(Digit(img, label))
            progress += 1
        print("Loaded!")
        return train_set, test_set
    else:
        for img, label in test_dataset:
            if progress == max_size:
                break
            train_set.add(Digit(img, label))
            progress += 1
        print("Loaded!")
        return train_set
    
