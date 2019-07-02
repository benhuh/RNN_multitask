""" Network model that works with train_pytorch.py """

import torch
from torch import nn, jit
from .train import generate_trials as gen_trials

class Model(nn.Module):
    def __init__(self, hp, RNNLayer): 
        super().__init__()
        n_input, n_rnn, n_output, decay = hp['n_input'], hp['n_rnn'], hp['n_output'], hp['decay']
        
        if hp['activation'] == 'relu':    # Type of activation runctions, relu, softplus, tanh, elu
            nonlinearity = nn.ReLU()
        elif hp['activation'] == 'tanh': 
            nonlinearity = nn.Tanh()
        else: 
            raise NotImplementedError
        
        self.n_rnn = n_rnn
        self.rnn   = RNNLayer(n_input, n_rnn, nonlinearity, decay)
        self.readout = nn.Linear(n_rnn, n_output, bias = False)

    def forward(self, x):
        hidden0   = torch.zeros([1, x.shape[1], self.n_rnn])  # initial hidden state
        hidden, _ = self.rnn(x, hidden0)
        output    = self.readout(hidden)
        return output, hidden        
    

class Run_Model(nn.Module): #(jit.ScriptModule):
    def __init__(self, hp, RNNLayer):
        super().__init__()
        self.hp = hp
        self.model = Model(hp, RNNLayer)
        self.loss_fnc = nn.MSELoss() if hp['loss_type'] == 'lsq' else nn.CrossEntropyLoss()

    def generate_trials(self, rule, hp, mode, batch_size):
        return gen_trials(rule, hp, mode, batch_size)
    
    def calculate_loss(self, output, hidden, trial, hp):
        loss     = self.loss_fnc(trial.c_mask * output, trial.c_mask * trial.y)
        loss_reg = (hidden.abs().mean() * hp['l1_h'] + hidden.norm() * hp['l2_h'])  #    Regularization cost  (L1 and L2 cost) on hidden activity

        for param in self.parameters():
            loss_reg += param.abs().mean() * hp['l1_weight'] + param.norm() * hp['l2_weight']   #    Regularization cost  (L1 and L2 cost) on weights
        return loss, loss_reg
    
#     @jit.script_method
    def forward(self, rule, batch_size = None, mode = 'random'): #, **kwargs):
        hp             = self.hp        
        trial          = self.generate_trials(rule, hp, mode, batch_size)
        output, hidden = self.model(trial.x)
        loss, loss_reg = self.calculate_loss(output, hidden, trial, hp)
        return loss, loss_reg, output.detach().cpu().numpy(), hidden.detach().cpu().numpy(), trial
    
    