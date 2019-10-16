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
from multiprocessing import Pool
from random_stepper import gather_experience
from datetime import datetime
import pickle


# both mu and logvar have shape (batch_size, 32)
def kl_divergence(mu, logvar):
    kls = 0.5*(-logvar - 1 + logvar.exp() + mu.pow(2)) # (batch_size, 32)
    kls = kls.sum(dim=1) # kl divergence for each sample. (batch_size, )
    
    avg_kl = kls.mean()
    return avg_kl

# class that interacts with the vae model during training
class Trainer():
    def __init__(self):
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.beta = 10
        self.epochs = 1
        
        self.num_workers = 32
        self.num_rollouts =  10000
        self.minibatch_size = 512
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare for vae training
        self.beta_vae = BetaVAE().to(self.device)
        self.optimizer = optim.Adam(self.beta_vae.parameters())
        self.loss_hist = []


    # get random rollouts in environment
    def get_experience(self):
        rollout_idxs = range(self.num_rollouts)

        with Pool(processes = self.num_workers) as pool: # create pool of workers
            pool.map(gather_experience, rollout_idxs, chunksize=1) # let each gather experience


    # unsupervised training of vae on random rollouts collected by get_experience
    def train_vae(self):
        self.beta_vae.train() # for dropout, batchnorm, and the like

        data_dir = "data/rollouts/"
        for filename in os.listdir(data_dir): # for each rollout saved on disk
            print("new file! #######################")

            with open(data_dir+filename, "rb") as data:  # load it into RAM
                self.replay_buffer = pickle.load(data)

            sampler = self.minibatch_sampler() # generate minibatches from it
        
            for minibatch in sampler: # train on each minibatch
                print("new minibatch")
                self.minibatch = minibatch
                self.train_step()

            break   # stopping after one file for testing


    # takes list of np arrays in self.replay_buffer and groups into minibatches
    def minibatch_sampler(self, shuffle=True):
        actions, observations = zip(*self.replay_buffer)
        num_minibatches = int(np.ceil(len(self.replay_buffer)/self.minibatch_size))

        for epoch in range(self.epochs):
            X = torch.tensor(observations).float() # tensor with dim 0 being the length of the original list
            
            if shuffle:
                shuffled_idxs = torch.tensor(np.random.permutation(len(self.replay_buffer)))
                X = torch.index_select(X, 0, shuffled_idxs) # select the rows of X in shuffled order
            
            X = torch.chunk(X, num_minibatches, dim=0) # now a list of tensors

            for minibatch in X:
                yield minibatch


    # defines loss and takes one step of gradient descent to minimize loss
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


    # saves vae model parameters
    def save_model(self, timestr):
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        torch.save(self.beta_vae.state_dict(), models_dir+"/model_"+timestr+".pt")


    # loads vae model parameters and prepares for inference
    def load_model(self, path):
        self.beta_vae.load_state_dict(torch.load(path))
        self.beta_vae.eval()


# get a string representing the current time
def get_time():
    timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return timestr


# assumes that get_experience has already been called
if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_vae()

    timestr = get_time() # get single time string so that model files and plot files will be named consistently

    trainer.save_model(timestr)
    trainer.plot_loss(timestr)
