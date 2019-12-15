from torch import nn, tensor
import torch
import torch.nn.functional as F
from Base_RNN import Base_RNN

class CTC_Model(Base_RNN):

    def __init__(self, vocab_size, hidden_layer = 256, drop_out = 0.5, n_layers = 2, cell_type = "lstm"):
        super(CTC_Model, self).__init__(vocab_size,hidden_layer, 0, drop_out, n_layers, cell_type)
        self.inputs = vocab_size

        self.rnn = self.rnn_cell(vocab_size, hidden_layer, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, x):

        y_pred = self.rnn(x)

        return y_pred