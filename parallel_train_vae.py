# TODO: refactor into class

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
from random_stepper import gather_experience

mse_loss = nn.MSELoss()
training_steps = 100
beta = 10

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
    # setup workers
    num_workers = os.cpu_count() # not sure of the performance implications of using all cores on workers, not leaving the main process one to itself
    pipes = [Pipe(duplex=True) for worker in range(num_workers)] # one two-way pipe per worker
    child_conns, parent_conns = zip(*pipes)
    seeds = range(1234, 1234+num_workers) # arbitrary positive integers
    process_args = zip(child_conns, seeds) # an iterable that yields one tuple of arguments per task/worker

    # prepare for vae training
    beta_vae = BetaVAE()
    optimizer = optim.Adam(beta_vae.parameters())
    loss_hist = np.zeros((2, training_steps))

    with Pool(processes = num_workers) as pool:
        pool.map_async(gather_experience, process_args, chunksize=1)

        while True:
            messages = [] # TODO: replace with replay buffer
            for conn in parent_conns:
                messages.append(conn.recv()) # blocks until there is something to receive or the connection is closed
            # TODO: sample from replay buffer
            train_step(minibatch)

    return loss_hist

def train_step(x):
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


loss_hist = train_vae()
plt.plot(range(training_steps), loss_hist[0], label="Recon Loss")
plt.plot(range(training_steps), loss_hist[1], label="KL Loss")
plt.plot(range(training_steps), np.sum(loss_hist, 0), label="Total Loss")
plt.legend()
plt.show()
