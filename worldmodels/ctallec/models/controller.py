import torch
import torch.nn as nn
from worldmodels.box_carry_env import BoxCarryEnv

class Controller(nn.Module):
    def __init__(self, latent_size, rnn_size, num_agents):
        super().__init__()
      
        self.actors = nn.ModuleList()
        for _ in range(num_agents):
            self.actors.append(nn.Linear(latent_size + rnn_size, BoxCarryEnv.action_space.n))
        self.softmax = nn.Softmax(dim=-1)

        self.critic = nn.Linear(latent_size + rnn_size, 1)

    def forward(self, vae_latents, rnn_hiddens):
        x = torch.cat([vae_latents, rnn_hiddens], dim=1)
        action_probs = [self.softmax(actor(x)) for actor in self.actors]
        state_values = self.critic(x)

        return action_probs, state_values
