import torch
import torch.nn as nn

class VariationalEncoder(nn.Module):
    def __init__(self, input_shape=5, latent_dim=2):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=input_shape, out_features=latent_dim
        )
        self.linear2 = nn.Linear(
            in_features=latent_dim, out_features=latent_dim
        )

        self.linear3 = nn.Linear(
            in_features=latent_dim, out_features=latent_dim
        )
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, features):
        features = torch.relu(self.linear1(features))
        mu = self.linear2(features)
        log_sigma = self.linear3(features)
        sigma = torch.exp(log_sigma)
        self.N.loc = self.N.loc.to(features.device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(features.device)

        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - 2*log_sigma - 1).sum()
        return z

class VariationalDecoder(nn.Module):
    def __init__(self, output_shape=5, latent_dim=2):
        super().__init__()
        self.hidden_layer = nn.Linear(
            in_features=latent_dim, out_features=latent_dim
        )
        self.output_layer = nn.Linear(
            in_features=latent_dim, out_features=output_shape
        )

    def forward(self, features):
        activation = self.hidden_layer(features)
        activation = torch.relu(activation)
        reconstructed = self.output_layer(activation)
        return reconstructed

class VAE(nn.Module):
    def __init__(self, input_shape=5, latent_dim=2):
        super().__init__()
        self.encoder = VariationalEncoder(input_shape=input_shape, latent_dim=latent_dim)
        self.decoder = VariationalDecoder(output_shape=input_shape, latent_dim=latent_dim)

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed
