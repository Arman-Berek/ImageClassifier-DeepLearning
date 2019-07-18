import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict
import os, random

import helper

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action = 'store', help='image directory')
parser.add_argument('--save_dir', action = 'store', help = 'save checkpoint')
parser.add_argument('--in_file', type = str, default = "label_map.json", help = 'input json file')
parser.add_argument('--arch', action = 'store',  default = 'vgg19', help = 'architecture')
parser.add_argument('--epochs', action = 'store',  type = int, default = 6, help = 'number of epochs')
parser.add_argument('--learning_rate', action = 'store', type = float, default = 0.03, help = 'learning rate')
parser.add_argument('--hidden_units', action = 'store',  type = int, default = 2900, help = 'number of hidden units')
parser.add_argument('--out_size', action = 'store',  type = int, default = 102, help = 'number of outputs')
parser.add_argument('--drop_p', type = float, default = 0.5, help = 'probability of dropping the weights')
parser.add_argument('--gpu', action = 'store_true', help = 'use gpu')
args = parser.parse_args()
json_path = args.in_file
label_map = helper.load_label_map(json_path)
data_dir = args.data_dir
train_data, validation_data, test_data, trainloader, validloader, testloader = helper.preprocess(data_dir)
model_ = classifier.build_model(args.hidden_units, len(label_map), args.drop_p, args.arch)
model = helper.premodel(args.arch)

for param in model.parameters():
    param.requires_grad = False
in_size = helper.get_size(model, args.arch)
model.classifier = helper.Network(in_size, args.out_size, [args.hidden_units], drop_p = 0.5)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr = args.learning_rate)
helper.train(model, trainloader, validloader, criterion, optimizer, args.epochs, 40, args.gpu)
test_accuracy, test_loss = helper.valid_loss_acc(model, testloader, criterion, args.gpu)
print("Test Accuracy: {:.4f} ".format(test_accuracy), "Test Loss: {:.4f}".format(test_loss))
helper.save_checkpoint(model, train_data, optimizer, args.save_dir, args.arch)
