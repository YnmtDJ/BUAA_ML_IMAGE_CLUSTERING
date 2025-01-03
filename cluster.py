import json
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import ImageDataset
from dataset.transform import create_transforms
from module.auto_encoder import SimpleAutoEncoder


def get_features(image_size, data_dir, batch_size, checkpoint_path, device):
    """
    Extract features from the model.
    """
    _, test_transform = create_transforms(image_size)
    dataset = ImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleAutoEncoder().to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    features = {}
    with torch.no_grad():
        for img, file_name in tqdm(dataloader):
            img = img.to(device)
            img = test_transform(img)
            out, feature = model(img)

            for i in range(len(file_name)):
                features[file_name[i]] = feature[i].cpu().numpy().tolist()

    return features


def cluster(features, cluster_num):
    """
    Cluster the features by KMeans
    """
    data = np.array([features[key] for key in features])

    # cluster the features by KMeans
    kmeans = KMeans(n_clusters=cluster_num, init='k-means++', random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_

    cluster_result = {key: int(label) for key, label in zip(features.keys(), labels)}

    # count the number of each label
    label_counts = Counter(cluster_result.values())
    print(f"Label counts: {label_counts}")

    return cluster_result


if __name__ == '__main__':
    # Parameters
    batch_size = 256
    cluster_num = 2
    image_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = "../images/"
    checkpoint_path = "./checkpoint/autoencoder.pth"
    result_path = "./cluster_result.json"

    features = get_features(image_size, data_dir, batch_size, checkpoint_path, device)
    cluster_result = cluster(features, cluster_num)

    with open(result_path, 'w') as f:
        json.dump(cluster_result, f)
