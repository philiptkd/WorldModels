import torch
import torch.nn as nn
from worldmodels.vae.vae import View
import numpy as np

latent_size = 32 # architecture dependent. TODO: refer to variable in vae model instead
action_size = 3 # environment dependent. TODO: don't hardcode this
input_size = latent_size + action_size
hidden_size = 256 # as per the World Models paper
num_gaussians = 5 # as per the World Models paper
gaussian_size = 1+latent_size*2 # Pi is a scalar, and both mu and logvar have length of latent_size
gmm_output_size = num_gaussians*gaussian_size # output = [(Pi, mu, logvar)_1, ..., (Pi, mu, logvar)_5]. mu and logvar parameterize a normal distribution over latent observations
split_sections = [1, latent_size, latent_size]

class MDN_RNN(nn.Module):
    def __init__(self):
        super(MDN_RNN, self).__init__()

        self.temperature = 1.15 # may need to be tuned. the paper isn't clear

        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.gmm = nn.Sequential(
            nn.Linear(hidden_size, gmm_output_size),
            View((-1, num_gaussians, gaussian_size))
            )
        self.softmax = nn.Softmax(dim=1)

        self.init_hidden_states()

    def init_hidden_states(self):
        self.h = torch.zeros((1, hidden_size))
        self.c = torch.zeros((1, hidden_size))


    # takes input x with shape (batch_size, input_size)
    # h, and c all have shape (batch_size, hidden_size)
    def forward(self, x):
        self.h, self.c = self.rnn(x, (self.h, self.c))

        y = self.gmm(self.h) # (batch_size, num_gaussians, gaussian_size)
        Pis, mus, logvars = torch.split(y, split_sections, dim=2) # each of shape (batch_size, num_gaussians, -1)
        Pis = self.softmax(Pis.squeeze(dim=2)) # (batch_size, num_gaussians)

        z = self.reparameterize(Pis, mus, logvars) # (batch_size, latent_size)

        return z

    # sampling with reparameterization trick
    def reparameterize(self, Pis, mus, logvars):
        # apply temperature to increase stochasticity
        Pis = Pis.div(self.temperature)
        logvars += np.log(self.temperature) # same as multiplying variance by temperature
        
        # pick a gaussian for each example in batch by sampling from categorical distribution in Pis
        gaussian_idxs = torch.multinomial(Pis, num_samples=1).squeeze(dim=1) # (batch_size, )

        # make 2d matrix, where each row is a parameter
        mus = mus.reshape((-1, latent_size)) 
        logvars = logvars.reshape((-1, latent_size))

        # adapt indices to reshaped tensors
        batch_size = gaussian_idxs.shape[0]
        batch_offsets = torch.tensor(range(batch_size))*num_gaussians # (batch_size, ). [0, 5, 10, 15, ...]
        gaussian_idxs = gaussian_idxs.add(batch_offsets)
       
        # get parameters of selected gaussians
        mus = torch.index_select(mus, dim=0, index=gaussian_idxs) # (batch_size, latent_size)
        logvars = torch.index_select(logvars, dim=0, index=gaussian_idxs) # (batch_size, latent_size)

        # sample from selected gaussians
        std = logvars.div(2).exp()
        eps = torch.randn_like(std)
        return mus + std*eps # (batch_size, latent_size)
