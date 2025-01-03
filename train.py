import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import ImageDataset
from dataset.transform import create_transforms
from module.auto_encoder import SimpleAutoEncoder
from module.loss_function import ContrastiveLoss


if __name__ == '__main__':
    # Parameters
    batch_size = 256
    epochs = 300
    learning_rate = 0.0003
    image_size = 32
    temperature = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = "../images/"
    log_dir = "./log"
    checkpoint_dir = "./checkpoint"

    # dataset and dataloader
    train_transform, _ = create_transforms(image_size)
    dataset = ImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    autoencoder = SimpleAutoEncoder().to(device)
    autoencoder.train()

    # loss function and optimizer
    criterion_reconstruct = nn.BCELoss()
    criterion_contrastive = ContrastiveLoss(batch_size, temperature)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # log tool
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        sum_loss = {"reconstruct": 0., "contrastive": 0.}
        print(f'Epoch {epoch+1} begin:')
        for img, file_name in tqdm(dataloader):
            img = img.to(device)
            img_i = train_transform(img)
            img_j = train_transform(img)
            out_i, feature_i = autoencoder(img_i)
            out_j, feature_j = autoencoder(img_j)

            loss_reconstruct = 0.5*(criterion_reconstruct(out_i, img_i) + criterion_reconstruct(out_j, img_j))
            loss_contrastive = criterion_contrastive(feature_i, feature_j)
            loss = loss_reconstruct + loss_contrastive
            sum_loss["reconstruct"] += loss_reconstruct.item()
            sum_loss["contrastive"] += loss_contrastive.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:  # save model every 20 epochs
            torch.save(autoencoder.state_dict(), f'{checkpoint_dir}/autoencoder-{epoch + 1}.pth')

        # write log
        writer.add_scalar('Loss/Reconstruct', sum_loss["reconstruct"] / len(dataloader), epoch+1)
        writer.add_scalar('Loss/Contrastive', sum_loss["contrastive"] / len(dataloader), epoch+1)
        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Reconstruct Loss: {sum_loss["reconstruct"] / len(dataloader):.4f}, '
              f'Contrastive Loss: {sum_loss["contrastive"] / len(dataloader):.4f}, ')

    writer.close()
