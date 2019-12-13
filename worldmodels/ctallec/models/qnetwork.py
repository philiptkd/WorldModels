import torch
import torch.nn as nn
from worldmodels.box_carry_env import BoxCarryEnv

class QNetwork(nn.Module):
    def __init__(self, latent_size, rnn_size, num_agents):
        super().__init__()
      
        self.heads= nn.ModuleList()
        for _ in range(num_agents):
            self.heads.append(nn.Linear(latent_size + rnn_size, BoxCarryEnv.action_space.n))


    def forward(self, vae_latents, rnn_hiddens):
        x = torch.cat([vae_latents, rnn_hiddens], dim=1)
        action_vals = [head(x) for head in self.heads]

        return action_vals
