from worldmodels.dynamics_model.dynamics_model import MDN_RNN
from worldmodels.trainer import Trainer
import numpy as np
from datetime import datetime

class RNN_Trainer(Trainer):
    def __init__(self):
        super(RNN_Trainer, self).__init__()

        self.epochs = 20
        self.num_workers = 32
        self.minibatch_size = 20
        
        # prepare for vae training
        self.model = MDN_RNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    # yields minibatches of actions and latent observations from the current replay buffer / file
    # self.replay_buffer should already be populated by self.train()
    def minibatch_sampler(self):
        self.model.init_hidden_states() # reset h and c to zeros at start of each rollout

        mus, logvars, actions = self.replay_buffer # all 2D numpy arrays with same first dimension length (num steps in rollout = batch_size)
        batch_size = mus.shape[0]
        num_minibatches = int(np.ceil(batch_size/self.minibatch_size))

        # reparameterize / sample z from mus, logvars
        std = np.exp(logvars/2)
        eps = np.random.randn(*std.shape)
        z = mus + std*eps # numpy array (batch_size, latent_size)
      
        x = np.concatenate((z,actions), axis=1) # concatenate z and a to form inputs to RNN
        z_next = np.roll(z, -1, axis=1) # also yield z_next as labels for RNN
     
        # take off last step, since it has no ground truth z_next label
        x = x[:-1,:]
        z_next = z_next[:-1,:]

        # convert to tensors
        X = torch.tensor(x).float() # (batch_size, input_size)
        Z_next = torch.tensor(z_next).float() # (batch_size, latent_size)
        
        # chunk into minibatches
        X = torch.chunk(X, num_minibatches, dim=0)
        Z_next = torch.chunk(Z_next, num_minibatches, dim=0)

        for minibatch in zip(X, Z_next):
            yield minibatch

    
    def train_step(self):
        x, z_next = self.minibatch
        x = x.to(self.device) # (seq_len, input_size). usually seq_len = minibatch_size
        z_next = z_next.to(self.device) # (seq_len, latent_size)

        assert x.shape[0] == z_next.shape[0]
        seq_len = x.shape[0]

        # prepare for new sequence
        loss = 0
        self.model.h.detach()
        self.model.c.detach()

        for i in range(seq_len):
            # index to get one step at a time, but give batch size 1
            this_x = x[i].unsqueeze(dim=0)
            this_z_next = z_next[i].unsqueeze(dim=0)

            z_pred = self.model(this_x)
            loss += self.loss_fn(z_pred, this_z_next) #TODO: check how appropriate the default loss fn is
        self.loss_hist.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def plot_loss(self):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = range(len(loss_hist))

        ax.plot(x, self.loss_hist, label="Total Loss")
        ax.legend()
        plt.title("MDN-RNN Training Loss")
        #plt.show()

        timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        plots_dir = "plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        fig.savefig(plots_dir+"/rnn_loss_plot"+timestr+".png")


# assumes that pre_encode has already been called
if __name__ == '__main__':
    data_dir = "vae/data/encoded/"
    trainer = RNN_Trainer()
    trainer.train(data_dir)

    trainer.save_model(timestr)
    trainer.plot_loss(timestr)
