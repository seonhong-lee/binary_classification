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
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)
    
    p.add_argument('--min_vocab_freq', type=int, default=5)
    p.add_argument('--max_vocab_size', type=int, default=999999)
    
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs',type=int, default=10)
    
    p.add_argument('--word_vec_size', type=int, default=256)
    p.add_argument('--dropout', type=float, default=.3)
    
    p.add_argument('--max_length', type=int, default=256)
    
    pass
