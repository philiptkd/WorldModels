import gym
from PIL import Image
import numpy as np
from vae import BetaVAE
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

import multiprocessing as mp
from multiprocessing import Pool

mse_loss = nn.MSELoss()
training_steps = 100
beta = 10

def preprocess(img_arr, new_size=(64, 64)):
    img = Image.fromarray(img_arr)
    img = img.resize(new_size)
    new_arr = np.asarray(img)/255.0 # (64, 64, 3)
    new_arr = np.transpose(new_arr, (2, 0, 1)) # (3, 64, 64)
    new_arr = np.expand_dims(new_arr, 0) # (1, 3, 64, 64)
    return torch.tensor(new_arr).float()

def reconstruction_loss(x, x_recon):
    batch_size = x.size(0)
    assert batch_size != 0

    return mse_loss(x, x_recon).div(batch_size)

# both mu and logvar have shape (batch_size, 32)
def kl_divergence(mu, logvar):
    kls = 0.5*(-logvar - 1 + logvar.exp() + mu.pow(2)) # (batch_size, 32)
    kls = kls.sum(dim=1) # kl divergence for each sample. (batch_size, )
    
    avg_kl = kls.mean()
    return avg_kl

def train_vae():
    env = gym.make("CarRacing-v0")
    env.reset()
    beta_vae = BetaVAE()
    optimizer = optim.Adam(beta_vae.parameters())

    loss_hist = np.zeros((2, training_steps))

    for i in range(training_steps):
        a = env.action_space.sample()
        obs, _, done, _ = env.step(a) # obs is 96 x 96 x 3 array of ints
        x = preprocess(obs) # (1, 3, 64, 64)
        
        x_recon, mu, logvar = beta_vae(x)

        recon_loss = reconstruction_loss(x, x_recon) 
        kl_loss = beta*kl_divergence(mu, logvar)
        loss = recon_loss + kl_loss

        loss_hist[0, i] = recon_loss.item()
        loss_hist[1, i] = kl_loss.item()
        print(i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            env.reset()

    env.close()

    return loss_hist

loss_hist = train_vae()
plt.plot(range(training_steps), loss_hist[0], label="Recon Loss")
plt.plot(range(training_steps), loss_hist[1], label="KL Loss")
plt.plot(range(training_steps), np.sum(loss_hist, 0), label="Total Loss")
plt.legend()
plt.show()
