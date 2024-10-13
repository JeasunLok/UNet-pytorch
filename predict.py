import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import os
import torch
from tqdm import tqdm
from utils.dataloader import *
from utils.utils import *
from model.unet_model import *

if __name__ == "__main__":
    # 是否使用GPU
    Cuda = True
    num_classes = 2
    predict_type = "Result" # ConfidenceInterval or Result
    model_path = r"checkpoints/2024-10-13-17-32-25/model_state_dict_loss0.1429_epoch18.pth"

    input_shape = [512, 512]
    output_folder = r"data/output"
    data_dir = r"data"

    if Cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device("cpu")

    print("===============================================================================")

    model = UNet(n_channels=3, n_classes=num_classes).to(device)    

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        # 加载权重
        state_dict = torch.load(model_path)
        
        # 移除 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith('module.') else k  # 去掉前缀
            new_state_dict[new_key] = v
        
        # 加载修改后的权重
        model.load_state_dict(new_state_dict)
        
    with open(os.path.join(data_dir, r"list/predict.txt"),"r") as f:
        predict_lines = f.readlines()
    num_predict = len(predict_lines)

    print("device:", device, "num_predict:", num_predict)
    print("===============================================================================")

    image_transform = get_transform(input_shape, IsResize=True, IsTotensor=True, IsNormalize=True)

    print("start predicting")

    model.eval()      
    for annotation_line in tqdm(predict_lines):
        name_image = annotation_line.split()[0]

        if name_image.endswith(".tif"):
            im_data, im_Geotrans, im_proj, cols, rows = read_tif(name_image)
        image = Image.open(name_image)
        name = os.path.basename(name_image)

        if image_transform is not None:
            image = image_transform(image)
            
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0 ,1]))

        image = image.unsqueeze(0)
        image = image.float().to(device)
        prediction = model(image)
        prediction = prediction.squeeze(0)
        prediction = F.softmax(prediction, dim=0)

        if predict_type == "Result":
            prediction = torch.argmax(prediction, dim=0)
        elif predict_type == "ConfidenceInterval":
            prediction = prediction * 100
            prediction = prediction.type(torch.uint8)
        else:
            raise ValueError("predict_type error!")

        if not os.path.exists(os.path.join(output_folder, name)):
            if name_image.endswith(".tif"):
                write_tif(os.path.join(output_folder, name), prediction.cpu().detach().numpy(), im_Geotrans, im_proj)
            else:
                cv2.imwrite(os.path.join(output_folder, name), prediction.cpu().detach().numpy())

                prediction_np = prediction.cpu().detach().numpy()
                vis_output_path = os.path.join(output_folder, name.split(".")[0] + '_visualization.png')
                prediction_min = prediction_np.min()
                prediction_max = prediction_np.max()

                # 避免除以0的情况
                if prediction_max != prediction_min:
                    prediction_normalized = ((prediction_np - prediction_min) / (prediction_max - prediction_min)) * 255
                else:
                    prediction_normalized = prediction_np

                prediction_normalized = prediction_normalized.astype('uint8')

                    # 转换为uint8类型
                prediction_normalized = prediction_normalized.astype('uint8')
                cv2.imwrite(vis_output_path, prediction_normalized)

    print("finish predicting successfully")
    print("===============================================================================") 
