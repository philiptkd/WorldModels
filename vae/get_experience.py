# collects rollouts of a random policy for vae and dynamics model training
from worldmodels.vae.train_vae import VAE_Trainer
trainer = VAE_Trainer()
trainer.get_experience()
