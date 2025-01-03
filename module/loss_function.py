import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        """
        Contrastive loss function.
        :param batch_size: batch size
        :param temperature: temperature parameter for contrastive loss
        """
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        """
        Mask the positive samples from the batch.
        :param batch_size: batch size
        """
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        """
        Calculate contrastive loss. z_i and z_j are the features from the same image with different augmentations.
        """
        batch_size = z_i.size(0)
        if batch_size == self.batch_size:
            N = 2 * self.batch_size
            z = torch.cat((z_i, z_j), dim=0)

            sim = torch.matmul(z, z.T) / self.temperature
            sim_i_j = torch.diag(sim, self.batch_size)
            sim_j_i = torch.diag(sim, -self.batch_size)

            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_samples = sim[self.mask].reshape(N, -1)

            labels = torch.zeros(N).to(positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss = self.criterion(logits, labels)
            loss /= N
        else:
            mask = self.mask_correlated_samples(batch_size)

            N = 2 * batch_size
            z = torch.cat((z_i, z_j), dim=0)

            sim = torch.matmul(z, z.T) / self.temperature
            sim_i_j = torch.diag(sim, batch_size)
            sim_j_i = torch.diag(sim, -batch_size)

            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_samples = sim[mask].reshape(N, -1)

            labels = torch.zeros(N).to(positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss = self.criterion(logits, labels)
            loss /= N
        return loss
