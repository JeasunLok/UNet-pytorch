import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
import os
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils.dataloader import *
from utils.utils import *
from model.unet_model import *
import torch.distributed as dist

if __name__ == "__main__":
    # 是否使用GPU
    Cuda = True
    num_workers = 4
    distributed = False
    num_classes = 2
    predict_type = "Result" # ConfidenceInterval or Result
    model_path = r""

    input_shape = [512, 512]
    output_folder = r""
    data_dir = r""

    
    # 设置用到的显卡
    ngpus_per_node  = torch.cuda.device_count()
    if Cuda:
        if distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            init_ddp(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            local_rank = 0
    else:
        device = torch.device("cpu")
        local_rank = 0

    if local_rank == 0:
        print("===============================================================================")

    model = UNet(n_channels=3, n_classes=num_classes)      

    if Cuda:
        if distributed:
            model = model.cuda(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model = model.cuda()

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model.load_state_dict(torch.load(model_path))  
        
    with open(os.path.join(data_dir, r"list/predict.txt"),"r") as f:
        predict_lines = f.readlines()
    num_predict = len(predict_lines)

    if local_rank == 0:
        print("device:", device, "num_predict:", num_predict)
        print("===============================================================================")

    image_transform = get_transform(input_shape, IsResize=False, IsTotensor=True, IsNormalize=True)

    if local_rank == 0:
        print("start predicting")

    model.eval()      
    for annotation_line in tqdm(predict_lines):
        name_image = annotation_line.split()[0]

        im_data, im_Geotrans, im_proj, cols, rows = read_tif(name_image)
        image = Image.open(name_image)
        name = os.path.basename(name_image)

        if image_transform is not None:
            image = image_transform(image)
            
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0 ,1]))

        image = image.unsqueeze(0)
        image = image.to(device).float()
        prediction, out, embedding = model(image)
        prediction = prediction.squeeze(0)
        prediction = F.softmax(prediction, dim=0)

        if predict_type == "Result":
            prediction = torch.argmax(prediction, dim=0)
        elif predict_type == "ConfidenceInterval":
            prediction = prediction * 100
            prediction = prediction.type(torch.uint8)
        else:
            raise ValueError("predict_type error!")

        if local_rank == 0:
            if not os.path.exists(os.path.join(output_folder, name)):
                write_tif(os.path.join(output_folder, name), prediction.cpu().detach().numpy(), im_Geotrans, im_proj, gdal.GDT_Byte)

    if distributed:
        dist.barrier()

    if local_rank == 0:
        print("finish predicting successfully")
        print("===============================================================================") 
