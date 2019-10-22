import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from datetime import datetime

class Trainer():
    def __init__(self):
        self.minibatch_size = 512
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.epoch = 0
        self.loss_hist = []
    
   
    def train(self):
        raise NotImplementedError


    def minibatch_sampler(self):
        raise NotImplementedError

    
    def train_step(self):
        raise NotImplementedError


    def plot_loss(self):
        raise NotImplementedError

    
    def get_time(self):
        timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        return timestr

    
    # saves vae model parameters
    #@profile
    def save_model(self, timestr):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_hist': self.loss_hist,
                    }, self.models_dir+"/model_"+timestr+".pt")


    # loads vae model parameters and prepares for inference
    def load_model(self, filepath=None):
        try:
            if filepath is None:
                model_files_list = os.listdir(self.models_dir)
                model_file = model_files_list[0]
                checkpoint = torch.load(self.models_dir+model_file)
            else:
                checkpoint = torch.load(filepath)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.loss_hist = checkpoint['loss_hist']
            self.model.eval()
        except:
            print("Failed to load model.")
