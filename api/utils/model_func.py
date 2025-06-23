import torch
from torchvision.models import resnet18
import torchvision.transforms as T

import json
import joblib
import os

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def load_classes():
    with open('utils/imagenet-simple-labels.json') as f:
        labels = json.load(f)
        return labels
    
def class_id_to_label(i):
    labels = load_classes()
    return labels[i]
    
def load_pt_model():
    model = resnet18()
    model.load_state_dict(
        torch.load(
            './weights/resnet18-weights.pth',
            map_location='cpu'
    ))
    model.eval()
    return model

def load_sklearn_model():
    clf = joblib.load('./weights/logreg.pkl')
    return clf

def transform_image(img):
    transforms = T.Compose([
        T.Resize((224,224)),
        T.CenterCrop(100),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    print(transforms(img).shape)
    return transforms(img).unsqueeze(0)