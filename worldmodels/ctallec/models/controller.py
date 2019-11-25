import torch
import torch.nn as nn
from worldmodels.box_carry_env import BoxCarryEnv

class TraceModule(nn.Module):
    def __init__(self, lamb):
        super().__init__()
        self.lamb = lamb
        self.eligibilities = []
    
    # from https://stackoverflow.com/questions/54734556/pytorch-how-to-create-an-update-rule-that-doesnt-come-from-derivatives
    def _update_grads_with_eligibility(self, delta, discount, ep_t, gamma):
        params = list(self.parameters())
        lamb = self.lamb
        eligibilities = self.eligibilities

        is_episode_just_started = (ep_t == 0)
        if is_episode_just_started:
            eligibilities.clear()
            for i, p in enumerate(params):
                if not p.requires_grad:
                    continue
                eligibilities.append(torch.zeros_like(p.grad, requires_grad=False))

        # eligibility traces
        for i, p in enumerate(params):
            if not p.requires_grad:
                continue

            eligibilities[i][:] = (gamma * lamb * eligibilities[i]) + (discount * p.grad)
            p.grad[:] = delta.squeeze() * eligibilities[i]


    # updates the gradient as a result of a change in delta
    # uses existing eligibility traces
    def _increase_delta(self, delta_delta):
        params = list(self.parameters())
        eligibilities = self.eligibilities
        
        for i, p in enumerate(params):
            if not p.requires_grad:
                continue

            p.grad[:] += delta_delta.squeeze() * eligibilities[i]




class Actor(TraceModule):
    def __init__(self, latent_size, rnn_size, num_agents, lamb):
        super().__init__(lamb)
      
        self.heads= nn.ModuleList()
        for _ in range(num_agents):
            self.heads.append(nn.Linear(latent_size + rnn_size, BoxCarryEnv.action_space.n))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, vae_latents, rnn_hiddens):
        x = torch.cat([vae_latents, rnn_hiddens], dim=1)
        action_probs = [self.softmax(head(x)) for head in self.heads]

        return action_probs




class Critic(TraceModule):
    def __init__(self, latent_size, rnn_size, lamb):
        super().__init__(lamb)
        self.fc = nn.Linear(latent_size + rnn_size, 1)


    def forward(self, vae_latents, rnn_hiddens):
        x = torch.cat([vae_latents, rnn_hiddens], dim=1)
        state_values = self.fc(x)

        return state_values
