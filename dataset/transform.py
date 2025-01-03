from torchvision import transforms


def create_transforms(size):
    """
    Create image transformations.
    :param size: Image output size
    :return: Train and test image transformations
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size),
    ])

    return train_transform, test_transform
