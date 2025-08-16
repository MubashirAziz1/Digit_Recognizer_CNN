from torchvision import transforms


def get_train_transforms():

    return transforms.Compose([
        transforms.RandomRotation(15),  # Random rotation for data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization values
    ])


def get_test_transforms():

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization values
    ]) 