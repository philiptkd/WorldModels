import torch
import torch.nn as nn
import numpy as np
import os
import pickle

class Trainer():
    def __init__(self):
        self.minibatch_size = 512
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_hist = []
        self.loss_fn = nn.MSELoss(reduction="mean")
    
    # takes list of np arrays in self.replay_buffer and groups into minibatches
    def minibatch_sampler(self, shuffle=True):
        actions, observations = zip(*self.replay_buffer)
        num_minibatches = int(np.ceil(len(self.replay_buffer)/self.minibatch_size))

        for epoch in range(self.epochs):
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


    # unsupervised training of vae on random rollouts collected by get_experience
    def train(self, data_dir):
        self.model.train() # for dropout, batchnorm, and the like

        for filename in os.listdir(data_dir): # for each rollout saved on disk
            print("new file! #######################")

            with open(data_dir+filename, "rb") as data:  # load it into RAM
                self.replay_buffer = pickle.load(data)

            sampler = self.minibatch_sampler() # generate minibatches from it
        
            for minibatch, _ in sampler: # train on each minibatch of observations
                print("new minibatch")
                self.minibatch = minibatch
                self.train_step()

    
    def train_step(self):
        raise NotImplementedError


    def plot_loss(self):
        raise NotImplementedError
    
    
    # saves vae model parameters
    def save_model(self, timestr):
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        torch.save(self.model.state_dict(), models_dir+"/model_"+timestr+".pt")


    # loads vae model parameters and prepares for inference
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


