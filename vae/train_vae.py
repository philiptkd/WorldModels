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
        
        self.num_rollouts = 1250
        self.steps_per_rollout = 100
        self.num_workers = os.cpu_count() # not sure of the performance implications of using all cores on workers, not leaving the main process one to itself
        self.final_replay_size = self.num_workers*self.num_rollouts*self.steps_per_rollout
        self.minibatch_size = self.num_workers*self.steps_per_rollout # 800 now
        
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

            num_received_recently = 0
            while self.replay_buffer is None or self.replay_buffer.shape[0] < self.final_replay_size:    
                for i,conn in enumerate(parent_conns):
                    ready = conn.poll()
                    if ready:
                        latest_experience = conn.recv()

                        if self.replay_buffer is None:
                            self.replay_buffer = latest_experience
                        else:
                            self.replay_buffer = torch.cat((self.replay_buffer, latest_experience)) # blocks until there is something to receive
                        print("received from worker",i)
                        num_received_recently += 1

                # only train VAE if enough new experience
                if num_received_recently >= self.num_workers//2: 
                    self.sample_minibatch()
                    self.train_step()
                    num_received_recently = 0
                    print("replay size:",self.replay_buffer.shape[0])

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
