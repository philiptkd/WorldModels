from worldmodels.dynamics_model.dynamics_model import MDN_RNN
from worldmodels.trainer import Trainer

class RNN_Trainer(Trainer):
    def __init__(self):
        super(RNN_Trainer, self).__init__()

        self.epochs = 20
        self.num_workers = 32
        
        # prepare for vae training
        self.model = MDN_RNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())


    def train_step(self):
        pass

    
    def plot_loss(self):
        pass


