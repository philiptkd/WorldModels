import gym
from PIL import Image
import numpy as np
from vae import BetaVAE
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os

import multiprocessing as mp
from multiprocessing import Pool, Pipe
from random_stepper import gather_experience

# both mu and logvar have shape (batch_size, 32)
def kl_divergence(mu, logvar):
    kls = 0.5*(-logvar - 1 + logvar.exp() + mu.pow(2)) # (batch_size, 32)
    kls = kls.sum(dim=1) # kl divergence for each sample. (batch_size, )
    
    avg_kl = kls.mean()
    return avg_kl

class Trainer():
    def __init__(self):
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.beta = 10
        self.minibatch_size = 512
        self.num_rollouts = 1

        self.num_workers = 1#os.cpu_count() # not sure of the performance implications of using all cores on workers, not leaving the main process one to itself
        self.seeds = range(self.num_workers) # arbitrary positive integers

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare for vae training
        self.beta_vae = BetaVAE().to(self.device)
        self.optimizer = optim.Adam(self.beta_vae.parameters())
        self.loss_hist = []
        self.replay_buffer = []
        self.done = False

    def train_vae(self):
        # setup workers
        pipes = [Pipe(duplex=True) for worker in range(self.num_workers)] # one two-way pipe per worker
        child_conns, parent_conns = zip(*pipes)
        process_args = zip(self.seeds, child_conns, [self.num_rollouts]*self.num_workers) # an iterable that yields one tuple of arguments per task/worker

        # create pool of workers
        with Pool(processes = self.num_workers) as pool:
            pool.map_async(gather_experience, process_args, chunksize=1)

            for rollout in range(self.num_rollouts):    
                # get message from each worker before continuing
                for i,conn in enumerate(parent_conns):
                        print(i, "waiting")
                        self.replay_buffer += conn.recv() # blocks until there is something to receive or the connection is closed
                        print(i, "received")

                #self.sample_minibatch()
                #self.train_step()
            print(self.replay_buffer)

    def sample_minibatch(self):
        idxs = np.randint(0, len(self.replay_buffer), size=self.minibatch_size)
        self.minibatch = torch.cat((obs for obs in self.replay_buffer[idxs]))

    def train_step(self):
        x = self.minibatch.to(device)
        x_recon, mu, logvar = self.beta_vae(x)

        recon_loss = self.loss_fn(x, x_recon)
        kl_loss = self.beta*kl_divergence(mu, logvar)
        loss = recon_loss + kl_loss

        self.loss_hist.append((recon_loss.item(), kl_loss.item()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_loss(self):
        recon_loss, kl_loss = zip(*self.loss_hist)

        plt.plot(recon_loss, label="Recon Loss")
        plt.plot(kl_loss, label="KL Loss")
        plt.plot(recon_loss + kl_loss, label="Total Loss")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_vae()
    #print(trainer.loss_hist)
