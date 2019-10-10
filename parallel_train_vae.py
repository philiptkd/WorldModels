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
        self.num_rollouts = 10
        self.steps_per_rollout = 100

        self.num_workers = os.cpu_count() # not sure of the performance implications of using all cores on workers, not leaving the main process one to itself
        self.seeds = range(self.num_workers) # arbitrary positive integers

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare for vae training
        self.beta_vae = BetaVAE().to(self.device)
        self.optimizer = optim.Adam(self.beta_vae.parameters())
        self.loss_hist = []
        self.replay_buffer = None
        self.done = False

    def train_vae(self):
        # setup workers
        pipes = [Pipe(duplex=True) for worker in range(self.num_workers)] # one two-way pipe per worker
        child_conns, parent_conns = zip(*pipes)
        process_args = zip(self.seeds, child_conns, [self.num_rollouts]*self.num_workers, [self.steps_per_rollout]*self.num_workers)

        # create pool of workers
        with Pool(processes = self.num_workers) as pool:
            pool.map_async(gather_experience, process_args, chunksize=1)

            for rollout in range(self.num_rollouts):    
                print("rollout", rollout)
                # get message from each worker before continuing
                for i,conn in enumerate(parent_conns):
                    
                    if self.replay_buffer is None:
                        self.replay_buffer = conn.recv()
                    else:
                        self.replay_buffer = torch.cat((self.replay_buffer, conn.recv())) # blocks until there is something to receive
                    print("received from worker",i)

                self.sample_minibatch()
                self.train_step()

    def sample_minibatch(self):
        idxs = torch.randint(0, self.replay_buffer.shape[0], size=(self.minibatch_size,))
        self.minibatch = torch.index_select(self.replay_buffer, 0, idxs)

    def train_step(self):
        x = self.minibatch.to(self.device)
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
    print(trainer.loss_hist)
