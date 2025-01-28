import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def create_dataloaders(data_dir: Path, batch_size=32, img_size=(224, 224)):
    """
    Creates DataLoaders for training, validation, and test datasets.

    Parameters:
    - data_dir (Path): Path to the dataset directory containing 'train', 'validation', and 'test' subdirectories.
    - batch_size (int): Number of samples per batch.
    - img_size (tuple): Size to resize images (default is 224x224).

    Returns:
    - train_loader, val_loader, test_loader: DataLoaders for train, validation, and test sets.
    """

    # Define a set of image transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ]),
        'validation': transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
    }

    # Load datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(root=data_dir / 'train', transform=data_transforms['train']),
        'validation': datasets.ImageFolder(root=data_dir / 'validation', transform=data_transforms['validation']),
        'test': datasets.ImageFolder(root=data_dir / 'test', transform=data_transforms['test'])
    }

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'validation': DataLoader(image_datasets['validation'], batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    return dataloaders['train'], dataloaders['validation'], dataloaders['test']
