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
    #@profile
    def __init__(self):
        super(VAE_Trainer, self).__init__()

        self.beta = 10
        self.epochs = 1
        self.num_workers = 24
        self.num_rollouts =  1000
        self.models_dir = "/home/teslaadmin/worldmodels/worldmodels/vae/models/"

        # prepare for vae training
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = BetaVAE().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.load_model() # load pre-trained weights if they exist

    # unsupervised training of vae on random rollouts collected by get_experience
    #@profile
    def train(self, data_dir, trained_rollouts_file=None):
        self.model.train() # for dropout, batchnorm, and the like

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.epochs):
            print(epoch)
            self.epoch = epoch
            filenames = os.listdir(data_dir)

            # TODO: remove this when memory allocation problems are resolved
            if trained_rollouts_file is not None:
                try:
                    with open(trained_rollouts_file,"r") as f:
                        trained_rollouts_list = [line.strip() for line in f]
                    filenames = [x for x in filenames if x not in trained_rollouts_list]
                except FileNotFoundError:
                    pass

            print(len(filenames))
            for filename in filenames: # for each rollout saved on disk
                with open(data_dir+filename, "rb") as data:  # load it into RAM
                    self.replay_buffer = pickle.load(data)

                sampler = self.minibatch_sampler() # generate minibatches from it

                for minibatch in sampler: # train on each minibatch of observations
                    self.minibatch = minibatch
                    self.train_step()
                    del minibatch, self.minibatch

                del self.replay_buffer
                print(filename+" done")

                # TODO: remove this when memory allocation problems are resolved
                if trained_rollouts_file is not None:
                    with open(trained_rollouts_file,"a+") as f:
                        f.write(filename+'\n')

                if not os.path.exists("models"):
                    os.makedirs("models")
                for f in os.listdir("models"): # remove all old models
                    os.remove("models/"+f)
                self.save_model(self.get_time()) # save current model


    # get random rollouts in environment
    # TODO: move to separate file
    def get_experience(self):
        # create data directory
        data_dir = "data/rollouts/"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        rollout_idxs = range(self.num_rollouts)
        rollout_idxs = [idx for idx in rollout_idxs if not os.path.isfile(data_dir+"rollout"+str(idx)+".pkl")]

        with Pool(processes = self.num_workers) as pool: # create pool of workers
            pool.map(gather_experience, rollout_idxs, chunksize=1) # let each gather experience


    # takes list of np arrays in self.replay_buffer and groups into minibatches
    #@profile
    def minibatch_sampler(self, shuffle=True):
        actions, observations = zip(*self.replay_buffer)
        num_minibatches = int(np.ceil(len(self.replay_buffer)/self.minibatch_size))

        X = torch.tensor(observations).float() # tensor with dim 0 being the length of the original list
        A = torch.tensor(actions).float()

        if shuffle:
            shuffled_idxs = torch.tensor(np.random.permutation(len(self.replay_buffer)))
            X = torch.index_select(X, 0, shuffled_idxs) # select the rows of X in shuffled order
            A = torch.index_select(A, 0, shuffled_idxs)

        X = torch.chunk(X, num_minibatches, dim=0) # now a list of tensors
        A = torch.chunk(A, num_minibatches, dim=0)

        for x,a in zip(X,A):
            yield x, a

        del actions, observations, X, A

    # defines loss and takes one step of gradient descent to minimize loss
    #@profile
    def train_step(self):
        x, _ = self.minibatch
        x = x.to(self.device)
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

        plots_dir = "plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        fig.savefig(plots_dir+"/vae_loss_plot"+timestr+".png")



# assumes that get_experience has already been called
if __name__ == '__main__':
    data_dir = "data/rollouts/"
    trainer = VAE_Trainer()
    
    try:
        trainer.train(data_dir, "rollouts_trained_on.txt")
    except KeyboardInterrupt:
        pass

    timestr = trainer.get_time() # get single time string so that model files and plot files will be named consistently

    trainer.save_model(timestr)
    trainer.plot_loss(timestr)
