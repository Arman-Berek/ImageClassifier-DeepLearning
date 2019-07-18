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
def predict(image_path, model, topk=5, cuda = False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = helper.process_image(image)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    inputs = Variable(image, requires_grad=False)
    inputs = inputs.unsqueeze(0)
    if cuda:
        inputs = inputs.cuda()
        model.cuda()
    ps = torch.exp(model.forward(inputs))
    if cuda:
        ps.cuda()
    top_probs, top_labels = ps.topk(top_k)
    top_probs, top_labels = top_probs.data.cpu().numpy().squeeze(), top_labels.data.cpu().numpy().squeeze()
    idx_to_class = {v: key for key, v in model.class_to_idx.items()}
    if top_k == 1:
        top_classes = [idx_to_class[int(top_labels)]]
        top_probs = [float(top_probs)]
    else:
        top_classes = [idx_to_class[each] for each in top_labels]

    return top_probs, top_classes
    
parser = argparse.ArgumentParser()
parser.add_argument('in', action = 'store', help = 'image path')
parser.add_argument('checkpoint', action = 'store', help = 'model')
parser.add_argument('--top_k', action = 'store', type = int, default = 5, help = 'top probable classes')
parser.add_argument('--cat_names', action = 'store', help = 'map of classes and names')
parser.add_argument('--gpu', action = 'store_true', help = 'use gpu')

args = parser.parse_args()
image_path = args.in
model = load_checkpoint(args.checkpoint)
if args.gpu and torch.cuda.is_available():
        model.cuda()
model.eval()
top_probs, top_classes = predict(image_path, model, top_k=args.top_k, cuda = args.gpu)
if args.cat_names != None:
    class_to_name = helper.load_label_map(args.cat_names)
    top_classes = [class_to_name[each] for each in top_classes]

for name, prob in zip(top_classes, top_probs):
    print(f"{name}= {prob:.4f}")
