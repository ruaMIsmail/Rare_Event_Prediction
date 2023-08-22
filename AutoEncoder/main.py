from data_process import DataProcessor
from AutoEncoder.train import AutoencoderClassifier
from AutoEncoder.model import HyperParameter
import pandas as pd 

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V


params = HyperParameter()

csv_file_path = 'Base.csv'
train  = AutoencoderClassifier(csv_file_path, params.device, params.batch_size, params.epochs, params.encoding_dim)