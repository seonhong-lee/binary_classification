import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    
    def __init__(
        self,
        input_size,
        word_vector_size,
        n_classes,
        use_batch_norm=False,
        dropout_p=.5,
        window_sizes=[3, 4, 5],
        n_filters=[100, 100, 100]
    ):
        self.input_size = input_size
        self.word_vector_size = word_vector_size
        self.n_classes = n_classes
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        self.window_sizes = window_sizes
        self.n_filters = n_filters
        
        super().__init__()
        
        self.emb = nn.Embedding(input_size, word_vector_size)
        
        self.feature_extractors = nn.ModuleList()
        for window_size, n_filter in zip(window_sizes, n_filters):
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=n_filter,
                        kernel_size=(window_size, word_vector_size)
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(n_filter) if use_batch_norm else nn.Dropout(dropout_p)
                )
            )        
            
            
        self.generator = nn.Linear(sum(n_filters), n_classes)
        self.activation = nn.LogSoftmax(dim=-1)
        
    
    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vector_size)
        min_length = max(self.window_sizes)
        if min_length > x.size(1):
            pad = x.new(x.size(0), min_length - x.size(1), self.word_vector_size).zero_()
            x = torch.cat([x, pad], dim=1)
            
            x = x.unsqueeze(1)
            
            cnn_outs = []
            for block in self.feature_extractors:
                cnn_out = block(x)
                # |cnn_out| = (batch_size, n_filter, length - window_size + 1, 1)
                cnn_out = nn.functional.max_pool1d(
                    input=cnn_outs.squeeze(-1),
                    kernel_size=cnn_out.size(-2)
                ).squeeze(-1)
                # |cnn_out| = (batch_size, n_filter)
                cnn_outs += [cnn_out]
                
        cnn_outs = torch.cat(cnn_outs, dim=-1)
        # |cnn_outs| = (batch_size, sum(n_filters))
        y = self.activation(self.generator(cnn_outs))
        # |y| = (batch_size, n_classes)
        
        return y