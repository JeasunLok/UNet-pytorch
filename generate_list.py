import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from utils import *

train_percent = 0.7
val_percent = 0.1
test_percent = 0.2

data_path = 'data'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt for trainning, validating and testing in data folder.")
    segfilepath = os.path.join(data_path, "label")
    saveBasePath = os.path.join(data_path, "list")
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".tif") or seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)   
    train_num = int(num*train_percent)  
    val_num = int(num*val_percent)
    test_num = int(num*test_percent)

    print("Train size: {:d} | Val size: {:d} | Test size: {:d}".format(train_num, val_num, test_num))

    # shuffle the list
    random.shuffle(total_seg)

    train_list = total_seg[0:int(num*train_percent)]
    val_list = total_seg[int(num*train_percent):int(num*(train_percent+val_percent))]
    test_list = total_seg[int(num*(train_percent+val_percent)):int(num*(train_percent+val_percent+test_percent))]

    # create the list file for trainning, validating and testing in data folder
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w') 
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')  
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')  
    
    for i in train_list:  
        # linux should add replace("\\", "\\\\")
        ftrain.write(os.path.join(segfilepath, i).replace("\\", "\\\\") + '\n')
        # ftrain.write(os.path.join(segfilepath, i) + '\n')
    for i in val_list:  
        # linux should add replace("\\", "\\\\")
        fval.write(os.path.join(segfilepath, i).replace("\\", "\\\\") + '\n')
        # fval.write(os.path.join(segfilepath, i) + '\n')
    for i in test_list:  
        # linux should add replace("\\", "\\\\")
        ftest.write(os.path.join(segfilepath, i).replace("\\", "\\\\") + '\n')
        # ftest.write(os.path.join(segfilepath, i) + '\n')

    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Create list successfully.")

    print("Check datasets format, this may take a while.")
    classes_nums = np.zeros([256], np.int64)
    for i in tqdm(range(num)):
        name = total_seg[i]
        label_name = os.path.join(segfilepath, name)
        if not os.path.exists(label_name):
            raise ValueError("There is no label %s, please check whether the file exists or its format (tif or png) is right."%label_name)
        
        if label_name.endswith(".tif"):
            label, im_Geotrans, im_proj, cols, rows = read_tif(label_name)
        elif label_name.endswith(".png"):
            label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        
        if len(np.shape(label)) > 2:
            print("The shape of %s is %s, please checkout the dataset format."%(name, str(np.shape(label))))
        classes_nums += np.bincount(np.reshape(label, [-1]), minlength=256)
            
    print("Print the pixels Key and Value.")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("Label only includes 0 and 255.")
        print("Binary classification should only include 0 for background, 1 for target.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("Only background in the label, please checkout the dataset format.")