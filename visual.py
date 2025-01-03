import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import ImageDataset
from dataset.transform import create_transforms
from module.auto_encoder import SimpleAutoEncoder


if __name__ == '__main__':
    # Parameters
    batch_size = 256
    image_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = "../images/"
    checkpoint_path = "./checkpoint/autoencoder.pth"
    cluster_result_path = "./cluster_result.json"
    max_image_show = 10

    _, test_transform = create_transforms(image_size)
    dataset = ImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    autoencoder = SimpleAutoEncoder().to(device)
    autoencoder.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    autoencoder.eval()

    cnt = 0  # count the number of images shown
    features = []  # store the features of the images
    labels = []  # store the labels of the images
    with open(cluster_result_path, 'r') as f:
        cluster_result = json.load(f)

    with torch.no_grad():
        for img, file_name in tqdm(dataloader):
            img = img.to(device)
            img = test_transform(img)
            output, feature = autoencoder(img)

            img = img[0].permute(1, 2, 0).cpu().numpy()
            output = output[0].permute(1, 2, 0).cpu().numpy()
            features.append(feature.cpu().numpy())
            for i in range(len(file_name)):
                labels.append(cluster_result[file_name[i]])

            if cnt < max_image_show:  # show the reconstructed images
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(img)
                axes[0].axis('off')
                axes[0].set_title('Original Image')
                axes[1].imshow(output)
                axes[1].axis('off')
                axes[1].set_title('Reconstructed Image')
                plt.show()
                cnt += 1

    features = np.concatenate(features, axis=0)

    # visualize the features
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], s=1, c=labels)
    plt.title('2D Visualization of Features with Labels')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(label='Label')
    plt.show()
