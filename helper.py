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

def preprocess(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)

    return train_data, validation_data, test_data, trainloader, validloader, testloader

def build_model(hidden_units, out_size, drop_prob, arch='vgg19'):
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)        
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print (arch + " is not available")
    for param in model.parameters():
        param.requires_grad = False
    in_size = model.classifier[0].in_features
    nur_net = Network(in_size, out_size, hidden_units, drop_prob)
    model.classifier = nut_net
    return model
        
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p = 0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:]) 
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p = drop_p)
    
    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x)) 
            x = self.dropout(x) 
        x = self.output(x) 
        return F.log_softmax(x, dim = 1)

def train(model, trainloader, validloader, criterion, optimizer, epochs, print_every = 40, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"
    steps = 0
    model.to('device')
    model.train() 
    for e in range(epochs):
        running_loss=0
        for ii, (images,labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            steps += 1
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                accuracy, validation_loss = valid_loss_acc(model, validloader, criterion)      
                print("Epoch: {}/{} ".format(e+1, epochs),
                      "Training Loss: {:.4f} ".format(running_loss / print_every),
                      "Validation Loss: {:.4f} ".format(validation_loss),
                      "Validation Accuracy: {:.4f}".format(accuracy))
                running_loss = 0
            model.train()
            
def valid_loss_acc(model, loader, criterion, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"
    model.eval()
    accuracy = 0
    loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images) 
            ps = torch.exp(output)
            equals = (labels.data == ps.max(1)[1])
            accuracy += equals.type_as(torch.FloatTensor()).mean() 
            loss += criterion(output, labels)     
    return accuracy / len(loader), loss / len(loader)

def save_checkpoint(model, train_data, optimizer, save_dir, arch):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch': arch,
                  'in_size': model.classifier.hidden_layers[0].in_features,
                  'out_size': model.classifier.output.out_features,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'drop_p': model.classifier.dropout.p,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.classifier.state_dict(),
                  'class_to_idx': model.class_to_idx}
    if(save_dir != None): 
        torch.save(model_checkpoint, save_dir+'checkpoint.pth')
    else:
        torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(check_path, arch = 'vgg119', drop_prob = 0.5):
    checkpoint = torch.load(check_path)
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)        
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print (arch + " is not available")
    for param in model.parameters():
        param.requires_grad = False    
    model.class_to_idx = checkpoint['class_to_idx']
    nur_net = Network(25088,  len(model.class_to_idx), checkpoint['hidden_layers'], drop_prob)
    model.classifier = nur_net
    model.load_state_dict(checkpoint['state_dict'])   
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256), 
                                         transforms.CenterCrop(224), 
                                         transforms.ToTensor()])
    img = img_transforms(img)   
    img = np.array(img)    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (np.transpose(img, (1, 2, 0)) - mean) / std    
    img = np.transpose(img, (2, 0, 1))
    return img

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
    

def load_label_map(json_path):
    with open(json_path, 'r') as f:
        label_map = json.load(f)
    return label_map
