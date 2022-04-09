from unicodedata import bidirectional
from torch import dropout
import torch.nn as nn

class RNNClassifier(nn.Module):
    
    def __init__(
        self,
        input_size,
        word_vector_size,
        hidden_size,
        n_classes,
        n_layers=4,
        dropout_p=.3,
    ):
        self.input_size = input_size
        self.word_vector_size = word_vector_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes,
        self.n_layers = n_layers,
        self.dropout_p = dropout_p
        
        super().__init__()
        
        self.emb_layer = nn.Embedding(input_size, word_vector_size)
        self.rnn = nn.LSTM(
            input_size=word_vector_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True,
        )
        self.generator = nn.Linear(hidden_size*2, n_classes)
        self.activation = nn.LogSoftmax(dim=-1)
        
        
    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb_layer(x)
        # |x| = (batch_size, length, word_vector_size)
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size*2)
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)
        
        return y