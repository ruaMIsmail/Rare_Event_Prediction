import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable as V
from torch.optim import lr_scheduler
import torch.nn.utils as utils

class HyperParameter:
    def __init__(self):
        self.batch_size = 500
        self.epochs = 100
        self.encoding_dim = 64
        self.hidden_dim = int(self.encoding_dim / 2)
        self.learning_rate = 1e-3
        self.input_dim = 38
        self.out_dim = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

params = HyperParameter()
class AE(nn.Module):
    def __init__(self, params):
        super(AE, self).__init__()

        self.params = params
        #create the layers encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.params.input_dim, self.params.encoding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.params.encoding_dim),  # Add Layer Normalization
        )

        #init encoder weights
        #for layer in self.encoder:
         #   if isinstance(layer, nn.Linear):
          #      torch.nn.init.kaiming_uniform_(layer.weight)

        # Create the layers for the decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.params.encoding_dim, self.params.input_dim),
            nn.LayerNorm(self.params.input_dim),  # Add Layer Normalization

        )

        # Create the layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(self.params.encoding_dim, self.paramsout_dim),
            nn.LayerNorm(self.params.out_dim),  # Add Layer Normalization
            nn.Dropout(0.2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification_output = self.classifier(encoded)
        return decoded, classification_output

