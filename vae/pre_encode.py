from worldmodels.vae.train_vae import VAE_Trainer
import os
import pickle
import torch
import numpy as np
from multiprocessing import Pool

# takes observations from random rollouts and encodes them with the encoder half of a (pretrained) vae
def encode_rollout(rollout_file):
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
    actions_list = []
    for obs, acts in sampler:
        x = obs.to(trainer.device)
        mu, logvar = trainer.beta_vae._encode(x)
        
        mu = mu.to(torch.device("cpu")).detach().numpy() # convert back to numpy arrays
        logvar = logvar.to(torch.device("cpu")).detach().numpy()
        latents_list.append((mu, logvar))
        actions_list.append(acts)

    # organize rollout encodings
    mus, logvars = zip(*latents_list) # separate into two different lists
    mus = np.concatenate(mus) # make into one big array
    logvars = np.concatenate(logvars)
    actions = np.concatenate(actions_list)
    params = [mus, logvars, actions]

    # write all encodings to file in data/encoded/ with name corresponding to rollout
    encoded_file_path = get_encoded_path(rollout_file)
    with open(encoded_file_path, "wb") as f:
        pickle.dump(params, f)
    print(rollout_file)
    

# return whether this rollout has been encoded and saved already
def is_encoded_already(rollout_file):
    encoded_file_path = get_encoded_path(rollout_file)
    return os.path.isfile(encoded_file_path)


# get the path of the encoded file for this rollout if it exists
def get_encoded_path(rollout_file):
    start_idx = len("rollout")
    end_idx = len(rollout_file) - len(".pkl")
    
    rollout_number = rollout_file[start_idx:end_idx]
    return encoded_dir+"encoding"+rollout_number+".pkl"


if __name__ == '__main__':
    # ensure the target directory exists
    encoded_dir = "data/encoded/"
    if not os.path.exists(encoded_dir):
        os.makedirs(encoded_dir)
    
    # select only those files that haven't been encoded yet
    rollout_dir = "data/rollouts/"
    rollout_files = os.listdir(rollout_dir)
    if len(rollout_files) == 0:
        raise FileNotFoundError("Could not find rollout data.")
    rollout_files = [x for x in rollout_files if not is_encoded_already(x)]

    with Pool(processes = 24) as pool: # create pool of workers
        pool.map(encode_rollout, rollout_files, chunksize=1) # let each gather experience

