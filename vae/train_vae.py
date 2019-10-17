import gym
from PIL import Image
import numpy as np
from worldmodels.vae.vae import BetaVAE
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os

import multiprocessing as mp
from multiprocessing import Pool
from worldmodels.vae.random_stepper import gather_experience
from datetime import datetime
import pickle
from worldmodels.trainer import Trainer


# both mu and logvar have shape (batch_size, 32)
def kl_divergence(mu, logvar):
    kls = 0.5*(-logvar - 1 + logvar.exp() + mu.pow(2)) # (batch_size, 32)
    kls = kls.sum(dim=1) # kl divergence for each sample. (batch_size, )
    
    avg_kl = kls.mean()
    return avg_kl

# class that interacts with the vae model during training
class VAE_Trainer(Trainer):
    def __init__(self):
        super(VAE_Trainer, self).__init__()

        self.beta = 10
        self.epochs = 1
        self.num_workers = 32
        self.num_rollouts =  10000
        
        # prepare for vae training
        self.model = BetaVAE().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())


    # get random rollouts in environment
    # TODO: move to separate file
    def get_experience(self):
        rollout_idxs = range(self.num_rollouts)

        with Pool(processes = self.num_workers) as pool: # create pool of workers
            pool.map(gather_experience, rollout_idxs, chunksize=1) # let each gather experience


    # defines loss and takes one step of gradient descent to minimize loss
    def train_step(self):
        x = self.minibatch.to(self.device)
        x_recon, mu, logvar = self.model(x)

        recon_loss = self.loss_fn(x, x_recon)
        kl_loss = self.beta*kl_divergence(mu, logvar)
        loss = recon_loss + kl_loss

        self.loss_hist.append((recon_loss.item(), kl_loss.item()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # saves matplotlib figure of loss over the course of training
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

        plots_dir = "plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        fig.savefig(plots_dir+"/vae_loss_plot"+timestr+".png")


# get a string representing the current time
def get_time():
    timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return timestr


# assumes that get_experience has already been called
if __name__ == '__main__':
    data_dir = "data/rollouts/"
    trainer = VAE_Trainer()
    trainer.train(data_dir)

    timestr = get_time() # get single time string so that model files and plot files will be named consistently

    trainer.save_model(timestr)
    trainer.plot_loss(timestr)
