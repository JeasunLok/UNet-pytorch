import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from utils.utils import *

data_path = r'/home/ljs/UNet-pytorch/data/predict'
list_path = r'data'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt for trainning, validating and testing in data folder.")
    segfilepath = os.path.join(data_path, "predict")
    saveBasePath = os.path.join(list_path, "list")
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".tif") or seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)  
    predict_list = total_seg

    print("Predict size: {:d}".format(num))

    # create the list file for predicting
    fpredict = open(os.path.join(saveBasePath, 'predict.txt'), 'w')  
    
    for i in predict_list:  
        # linux should add replace("\\", "\\\\")
        fpredict.write(os.path.join(segfilepath, i).replace("\\", "\\\\") + '\n')
        # ftrain.write(os.path.join(segfilepath, i) + '\n')

    fpredict.close()  
    print("Create list successfully.")