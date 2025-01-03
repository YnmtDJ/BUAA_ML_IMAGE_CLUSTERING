import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        """
        Dataset to load images.
        :param image_dir: The directory containing images.
        """
        self.image_dir = image_dir
        self.transform = transforms.ToTensor()
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, os.path.basename(image_path)
