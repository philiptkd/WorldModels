""" Define controller """
import torch
import torch.nn as nn
from worldmodels.box_carry_env import BoxCarryEnv

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.c1 = nn.Sequential(
                nn.Linear(latents + recurrents, BoxCarryEnv.action_space.n),
                nn.Softmax(dim=-1)
                )

        self.c2 = nn.Sequential(
                nn.Linear(latents + recurrents, BoxCarryEnv.action_space.n),
                nn.Softmax(dim=-1)
                )

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        logits = [self.c1(cat_in), self.c2(cat_in)]
        actions = torch.cat([torch.multinomial(x,1) for x in logits], dim=1).float()
        return actions
