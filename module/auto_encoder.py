from torch import nn
from torch.nn import functional as F


class SimpleAutoEncoder(nn.Module):
    def __init__(self, num_layers=3, hidden_channels=128, feature_dim=128):
        """
        Reconstruct the input image and extract image features.
        :param num_layers: number of layers in the model
        :param hidden_channels: hidden layer channels in the model
        :param feature_dim: feature dimension of the output tensor
        """
        super(SimpleAutoEncoder, self).__init__()
        # the number of channels in each layer
        channels = [hidden_channels // 2 ** (num_layers - i) if i > 0 else 3 for i in range(num_layers+1)]
        self.encoder = Encoder(num_layers, channels)
        channels.reverse()
        self.decoder = Decoder(num_layers, channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, feature_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)

        hidden = self.encoder(x)
        out = self.decoder(hidden)

        pool_hid = self.pool(hidden).view(batch_size, -1)
        feature = F.normalize(self.projector(pool_hid), p=2, dim=1)

        return out, feature


class Encoder(nn.Module):
    def __init__(self, num_layers, channels):
        """
        Encoder model using convolution layers.
        :param num_layers: Number of convolution layers
        :param channels: Number of channels in each layer
        """
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1))
            self.encoder.append(nn.BatchNorm2d(channels[i + 1]))
            self.encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, num_layers, channels):
        """
        Decoder model using transpose convolution layers.
        :param num_layers: Number of transpose convolution layers
        :param channels: Number of channels in each layer
        """
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            self.decoder.append(nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1))
            self.decoder.append(nn.BatchNorm2d(channels[i + 1]))
            if i < num_layers - 1:
                self.decoder.append(nn.ReLU())
            else:
                self.decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        return self.decoder(x)
