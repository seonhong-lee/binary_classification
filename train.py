import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from trainer import Trainer
from data_loader import DataLoader

from models.rnn import RNNClassifier
from models.cnn import CNNClassifier

def select_argparser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    
    pass