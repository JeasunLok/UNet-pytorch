import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
from utils import *
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, image_transform=None, label_transform=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name_label = annotation_line.split()[0]
        name_image = name_label.replace("label", "image").split(".")[0] + ".png"

        image = Image.open(name_image)
        label = Image.open(name_label)

        if label.mode != 'L':
            label = label.convert('L')

        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0 ,1]))
        if self.label_transform is not None:
            label = self.label_transform(label)
        label= torch.from_numpy(np.array(label))
        
        return image, label

    def __len__(self):
        return self.length
