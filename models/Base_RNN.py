from torch import nn, tensor
import torch
import torch.nn.functional as F

class Base_RNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, cell_type = "RNN"):
        super(Base_RNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_layer = hidden_size
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p = self.input_dropout_p)
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        if cell_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif cell_type.lower() == 'gru':
            self.rnn_cell = nn.GRU
        elif cell_type.lower() == 'rnn':
            self.rnn_cell = nn.RNN
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(cell_type))
        

        

    def forward(self, *args, **kwargs):
        raise NotImplementedError()