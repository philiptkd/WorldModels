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
from datetime import datetime
import pickle


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
        self.epochs = 1
        
        self.num_workers = 32#os.cpu_count() # not sure of the performance implications of using all cores on workers, not leaving the main process one to itself
        self.num_rollouts = self.num_workers
        self.minibatch_size = 512
        
        self.seeds = range(self.num_workers) # arbitrary positive integers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare for vae training
        self.beta_vae = BetaVAE().to(self.device)
        self.optimizer = optim.Adam(self.beta_vae.parameters())
        self.loss_hist = []


    def get_experience(self):
        buffer_filename = "replay_buffer.pkl"
        try:
            with open(buffer_filename, "rb") as f:
                self.replay_buffer = pickle.load(f)
            return
        except FileNotFoundError:
            print("Replay buffer not found. Populating it now. It may take a while.")
        
        worker_rollouts = self.num_rollouts//self.num_workers
        process_args = zip(self.seeds, [worker_rollouts]*self.num_workers)

        with Pool(processes = self.num_workers) as pool: # create pool of workers
            self.replay_buffer =  pool.map(gather_experience, process_args, chunksize=1) # let each gather experience

        with open("replay_buffer.pkl", "wb") as f:
            pickle.dump(self.replay_buffer, f)


    def train_vae(self):
        self.replay_buffer = self.get_experience() # list of (a, obs) tuples
        sampler = self.minibatch_sampler()
        
        for minibatch in sampler:
            self.minibatch = minibatch
            self.train_step()


    def minibatch_sampler(self):
        actions, observations = zip(*self.replay_buffer)
        num_minibatches = np.ceil(len(self.replay_buffer)/self.minibatch_size)

        for epoch in range(self.epochs):
            X = torch.tensor(observations).float() # tensor with dim 0 being the length of the original list
            shuffled_idxs = torch.tensor(np.random.permutation(len(replay_buffer)))
            X = torch.index_select(X, 0, shuffled_idxs) # select the rows of X in shuffled order
            X = torch.chunk(X, num_minibatches, dim=0) # now a list of tensors

            for minibatch in X:
                yield minibatch


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


    def plot_loss(self, timestr):
        recon_loss, kl_loss = zip(*self.loss_hist)
        total_loss = [recon_loss[i] + kl_loss[i] for i in range(len(kl_loss))]

        fig = plt.figure()
        ax = plt.subplot(111)
        x = range(len(recon_loss))

        ax.plot(x, recon_loss, label="Recon Loss")
        ax.plot(x, kl_loss, label="KL Loss")
        ax.plot(x, total_loss, label="Total Loss")
        ax.legend()
        ax.set_ylim(0, 0.2)
        plt.title("VAE Unsupervised Training Losses")
        #plt.show()

        timestr = get_time()
        fig.savefig("plots/vae_loss_plot"+timestr+".png")


    def save_model(self, timestr):
        torch.save(self.beta_vae.state_dict(), "models/model_"+timestr+".pt")


    def load_model(self, path):
        self.beta_vae.load_state_dict(torch.load(path))
        # don't forget to call model.eval() before inference


def get_time():
    timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return timestr


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_vae()

    timestr = get_time() # get single time string so that model files and plot files will be named consistently

    trainer.save_model(timestr)
    trainer.plot_loss(timestr)