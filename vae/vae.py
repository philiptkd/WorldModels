import torch
import torch.nn as nn

# implementing view as an nn.Module with forward() allows view to be an argument to nn.Sequential
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

# the beta-vae model
# built according to specifications in the appendix of Ha's World Models paper
class BetaVAE(nn.Module):
    def __init__(self):
        super(BetaVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2),
            View((-1, 2*2*256))
        )

        self.mu_dense = nn.Linear(2*2*256, 32)
        self.logvar_dense = nn.Linear(2*2*256, 32)

        self.decoder = nn.Sequential(
            nn.Linear(32, 1024),
            View((-1, 1024, 1, 1)),
            nn.ConvTranspose2d(1024, 128, 5, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 6, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        mu, logvar = self._encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z).view(x.size())
        
        return x_recon, mu, logvar

    def _encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu_dense(encoded)
        logvar = self.logvar_dense(encoded)
        return mu, logvar

    def _decode(self, z):
        return self.decoder(z)

    # reparameterization trick / sampling
    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.randn_like(std)
        return mu + std*eps

