from torch import nn, tensor
import torch
import torch.nn.functional as F
from models.Base_RNN import Base_RNN

class RNNLM(Base_RNN):

    def __init__(self, vocab_size, embed_size = 350, hidden_layer = 256, max_len = 100, input_dropout_p = 0.3, drop_out = 0.5, n_layers = 2, bidirectional = False , cell_type = "lstm"):
        super(RNNLM, self).__init__(vocab_size, hidden_layer, input_dropout_p, drop_out, n_layers, cell_type)
        
        self.bidirectional = bidirectional
        self.inputs = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.inputs + 1, self.embed_size, padding_idx= 0)
        self.rnn = self.rnn_cell(input_size = self.embed_size, hidden_size = self.hidden_layer, num_layers = self.n_layers,
                                 batch_first= True, bidirectional= self.bidirectional, dropout= self.dropout_p)
        self.max_len = max_len

        self.features = self.hidden_layer * self.max_len

        if self.bidirectional:
            self.linear = nn.Sequential(
                nn.Linear(self.features * 2, 1),
                nn.Sigmoid()
            )
        
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.features, 1),
                nn.Sigmoid()
            )

    def forward_step(self, x):
        batch_size = 0
        output_size = 0
        embedded = self.embedding(x)
        #embedded = self.input_dropout(embedded)

        if self.training:
            #train 일 경우 parameter 초기화
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, self.hidden_layer)
        
    
    def forward(self, x):

        x = self.embedding(x)
        x = self.input_dropout(x)
        
        output, (hidden, cell) = self.rnn(x)
        #make flatten (2dimension으로 차원 축소)
        output = output.reshape(output.size(0),-1)

        y_pred = self.linear(output)

        return y_pred