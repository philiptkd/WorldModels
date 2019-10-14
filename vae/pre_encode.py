# this could probably be made more efficient, but that's not a high priority right now

from train_vae import Trainer
import os
import pickle
import torch
import numpy as np
from multiprocessing import Pool

def encode_rollout(rollout_file):
    print(rollout_file)
    
    # load model
    trainer = Trainer()
    models_dir = "models/"
    model_files = os.listdir(models_dir)
    if len(model_files) == 0:
        raise FileNotFoundError("Could not find any saved beta-vae model files.")
    trainer.load_model(models_dir+model_files[0]) # just pick the first one

    # get a generator that yields minibatches from the rollout
    with open(rollout_dir+rollout_file, "rb") as f:
        trainer.replay_buffer = pickle.load(f)
    sampler = trainer.minibatch_sampler(shuffle=False)
    
    # encode to mu, std
    latents_list = []
    for minibatch in sampler:
        x = minibatch.to(trainer.device)
        mu, logvar = trainer.beta_vae._encode(x)
        
        mu = mu.to(torch.device("cpu")).detach().numpy() # convert back to numpy arrays
        logvar = logvar.to(torch.device("cpu")).detach().numpy()
        latents_list.append((mu, logvar))

    # organize rollout encodings
    mus, logvars = zip(*latents_list) # separate into two different lists
    mus = np.concatenate(mus) # make into one big array
    logvars = np.concatenate(logvars)
    params = [mus, logvars]

    # get rollout number
    start_idx = len("rollout")
    end_idx = len(rollout_file) - len(".pkl")
    rollout_number = rollout_file[start_idx:end_idx]

    # write all encodings to file in data/encoded/ with name corresponding to rollout
    encoded_dir = "data/encoded/"
    if not os.path.exists(encoded_dir):
        os.makedirs(encoded_dir)
    with open(encoded_dir+"encoding"+rollout_number+".pkl", "wb") as f:
        pickle.dump(params, f)


if __name__ == '__main__':
    rollout_dir = "data/rollouts/"
    rollout_files = os.listdir(rollout_dir)
    if len(rollout_files) == 0:
        raise FileNotFoundError("Could not find rollout data.")

    with Pool(processes = 24) as pool: # create pool of workers
        pool.map(encode_rollout, rollout_files, chunksize=1) # let each gather experience

