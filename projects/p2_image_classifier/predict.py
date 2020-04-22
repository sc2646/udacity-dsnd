import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import util


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    print("============= Image Processed =============")

        
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img.unsqueeze_(0)
    
    model.eval()
    
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = loaded_model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list

def main():
    parser = argparse.ArgumentParser(description='Predict a file.')
    parser.add_argument('input_img', type=str, help='Path of the input image', default="./flowers/test/2/image_05100.jpg")
    parser.add_argument('checkpoint', type=str, help='Path of the checkpoint', default="./checkpoint.pth")
    parser.add_argument ('--top_k', help = 'Top k', type = int, default=5)
    parser.add_argument ('--gpu', action='store_true', help='Use GPU for inference if GPU is available')
    parser.add_argument('--category_names', action="store", default='cat_to_name.json')

    args, _ = parser.parse_known_args()
    
    input_img = args.input_img
    checkpoint = args.checkpoint
    
    category_names = 'cat_to_name.json'
    if args.category_names:
        category_names = args.category_names
    
    top_k = 5
    if args.top_k:
        top_k = args.top_k
    
    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            print("GPU flag was set but no GPU is available in this machine.")
                
    loaded_model = util.load_checkpoint(checkpoint, cuda)
    
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
        
    probabilities, classes = util.predict(input_img, loaded_model, top_k)
        
    labels = [cat_to_name[str(int(index)+1)] for index in classes]
    probability = np.array(probabilities)

    i=0
    while i < top_k:
        print("{} with a probability of {:.2f}%".format(labels[i], probability[i]*100))
        i += 1

    print("Prediction Done.")

if __name__ == '__main__':
    main()
    