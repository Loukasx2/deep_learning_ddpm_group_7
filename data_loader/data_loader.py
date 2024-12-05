import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DatasetLoader:
    def __init__(self, dataset_name: str, batch_size: int = 64, data_dir: str = './data', shuffle: bool = True):
        """
        Initializes the dataset loader for MNIST or CIFAR-10.

        Args:
            dataset_name (str): Name of the dataset ('mnist' or 'cifar10').
            batch_size (int): Number of samples per batch. Default is 64.
            data_dir (str): Directory to store the dataset. Default is './data'.
            shuffle (bool): Whether to shuffle the data. Default is True.
        """
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.shuffle = shuffle

        # Initialize transformations and dataset
        self.transform = self._get_transform()
        self.train_dataset, self.test_dataset = self._get_dataset()

    def _get_transform(self):
        if self.dataset_name == 'mnist':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif self.dataset_name == 'cifar10':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _get_dataset(self):
        if self.dataset_name == 'mnist':
            train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=self.transform)
            test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=self.transform)
        elif self.dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform)
            test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return train_dataset, test_dataset

    def get_dataloader(self, train: bool = True):
        """
        Returns the dataloader for the specified dataset.

        Args:
            train (bool): Whether to return the training dataloader. Default is True.

        Returns:
            DataLoader: The PyTorch dataloader for the dataset.
        """
        dataset = self.train_dataset if train else self.test_dataset
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle if train else False)

# Example usage
if __name__ == "__main__":
    # MNIST example
    mnist_loader = DatasetLoader(dataset_name="mnist", batch_size=32)
    train_loader = mnist_loader.get_dataloader(train=True)
    test_loader = mnist_loader.get_dataloader(train=False)

    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"[MNIST] Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        break

    # CIFAR-10 example
    cifar_loader = DatasetLoader(dataset_name="cifar10", batch_size=32)
    train_loader = cifar_loader.get_dataloader(train=True)
    test_loader = cifar_loader.get_dataloader(train=False)

    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"[CIFAR-10] Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        break