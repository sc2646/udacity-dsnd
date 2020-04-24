import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from PIL import Image


def get_train_data(data_dir):
   train_dir = data_dir + '/train'
   # set train loader
   train_transforms = transforms.Compose([
                        transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])])
    
   # load dataset with ImageFolder
   train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
   return train_dataset

def get_dataloaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    print("============= Getting Data =============")

    # set train loader
    train_transforms = transforms.Compose([
                        transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])])
    
    # load dataset with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # set valid loader and test loader
    valid_transforms = transforms.Compose([
                        transforms.Resize(255),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([
                        transforms.Resize(255),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], 
                                             [0.229, 0.224, 0.225])])
    
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return trainloader, validloader, testloader




def nn_setup(arch, dropout, hidden_units, learning_rate, cuda):
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a valid model.".format(arch))
    
    structure = {"vgg16": 25088,
                  "densenet121" : 1024,
                  "alexnet" : 9216 }
   
    for param in model.parameters():
        param.requires_grad = False
        
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                    ('dropout',nn.Dropout(dropout)),
                    ('inputs', nn.Linear(structure[arch], hidden_units)),
                    ('relu1', nn.ReLU()),
                    ('hidden_layer1', nn.Linear(hidden_units, 90)),
                    ('relu2',nn.ReLU()),
                    ('hidden_layer2',nn.Linear(90, 80)),
                    ('relu3',nn.ReLU()),
                    ('hidden_layer3',nn.Linear(80, 102)),
                    ('output', nn.LogSoftmax(dim=1))
                                  ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    if torch.cuda.is_available() and cuda:
        model.cuda()
    
    return model, criterion, optimizer

def train_network(model, criterion, optimizer, epochs, print_every, train_loader, valid_loader, cuda):
    steps = 0
    print("============= Start Trainning Model =============")

    for e in range(epochs):
        running_loss = 0

        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            # Move input and label tensors to the GPU
            if torch.cuda.is_available() and cuda:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # zeroing parameter gradients
            optimizer.zero_grad()


            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Carrying out validation step
            if steps % print_every == 0:
                # setting model to evaluation mode during validation so that dropout is turned off
                model.eval()

                # Gradients are turned off as no longer in training
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion, cuda)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                      "Valid Accuracy: {:.3f}%".format(accuracy/len(valid_loader)*100))

                running_loss = 0

                # Turning training back on
                model.train()

def validation(model, validation_loader, criterion, cuda):
    valid_loss = 0
    accuracy = 0

    if torch.cuda.is_available() and cuda:
        model.to('cuda')

    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(validation_loader):
    
        # Change images and labels to work with cuda
        if torch.cuda.is_available() and cuda:
            images, labels = images.to('cuda'), labels.to('cuda')

        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)
        
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def save_model(model, arch, train_data, optimizer, save_dir, hidden_layer, dropout, lnr, epochs):    
    # Saving: feature weights, new classifier, index-to-class mapping, optimiser state, and No. of epochs
    checkpoint = {'state_dict': model.state_dict(),
                  'arch': arch, 
                  'classifier': model.classifier,
                  'mapping': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs,
                  'hidden_layer': hidden_layer,
                  'dropout': dropout, 
                  'learning_rate': lnr }

    print("============= Model Saved =============")
    return torch.save(checkpoint, save_dir + 'checkpoint.pth')

def load_checkpoint(save_dir, cuda):
    print("============= Loading Checkpoint =============")
    checkpoint = torch.load(save_dir)
    
    model, _, _ = nn_setup(checkpoint['arch'], checkpoint['dropout'], checkpoint['hidden_layer'], checkpoint['learning_rate'], cuda)
   
    model.classifier = checkpoint['classifier'] 
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    
    for param in model.parameters(): 
        param.requires_grad = False #turning off tuning of the model
   
    print("============= Checkpoint Loaded =============")
    return model

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print("============= Predicting Image... =============")

    model.cpu()    
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
        
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img.unsqueeze_(0)
    
    model.eval()
    
    with torch.no_grad():
        # Running image through network
        output = model.forward(img_add_dim)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_im = Image.open(image)

    # Building image transform
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    
    image = transform(pil_im)
        
    return image