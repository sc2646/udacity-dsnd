import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

import util

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Train a nn model')
    parser.add_argument('data_dir', type=str, help='Path of the image dataset', default="./flowers")
    parser.add_argument('--save_dir', help = 'Directory to save checkpoints', type = str)
    parser.add_argument('--arch', help = 'Default is alexnet, choose from alexnet, densenet121, or vgg16', type = str)
    parser.add_argument('--learning_rate', help = 'Learning rate', type = float)
    parser.add_argument('--hidden_units', help = 'Hidden units', type = int)
    parser.add_argument('--epochs', help = 'Epochs', type = int)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if GPU is available')
    parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)


    args, _ = parser.parse_known_args()
    
    data_dir = args.data_dir

    save_dir = './'
    if args.save_dir:
        save_dir = args.save_dir
    
    arch = 'alexnet'
    if args.arch:
        arch = args.arch

    learning_rate = 0.01
    if args.learning_rate:
        learning_rate = args.learning_rate

    hidden_units = 120
    if args.hidden_units:
        hidden_units = args.hidden_units

    epochs = 3
    if args.epochs:
        epochs = args.epochs
        
    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            print("GPU flag was set but no GPU is available in this machine.")
            
    dropout = args.dropout
    
    # Load the dataset
    trainloader, validloader, testloader = util.get_dataloaders(data_dir)

    
    model, optimizer, criterion = util.nn_setup(arch, dropout, hidden_units, learning_rate, cuda)
    
    util.train_network(model, optimizer, criterion, epochs, 40, trainloader, validloader, cuda)
    print("Trainning model done.")

    train_data = util.get_train_data(data_dir)
    util.save_model(model, arch, train_data, optimizer, save_dir, hidden_units, dropout, learning_rate, epochs)
    print("Trainned model saved.")

if __name__ == '__main__':
    main()