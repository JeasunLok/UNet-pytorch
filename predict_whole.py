import numpy as np
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
from utils.dataloader import *
from utils.utils import *
from model.unet_model import *

def sliding_window_predict(model, image, device, window_size=[512, 512], step_size=256, num_classes=2):
    """
    使用滑动窗口对大图进行模型预测。
    
    Args:
        model: 进行预测的模型
        image: 输入的大图 (4000多x3000多)
        device: 使用的设备 (CPU或GPU)
        window_size: 每次模型预测的图像块大小
        step_size: 滑动窗口的步长
        num_classes: 分类数
        
    Returns:
        全图的预测结果
    """
    height, width = image.shape[1], image.shape[2]
    patch_height, patch_width = window_size
    
    # 初始化预测结果的数组，和输入图像大小相同
    prediction_full = np.zeros((num_classes, height, width), dtype=np.float32)

    # 滑动窗口遍历大图
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            # 计算窗口的右边界和下边界
            y_end = min(y + patch_height, height)
            x_end = min(x + patch_width, width)

            # 提取当前窗口的图像块
            patch = image[:, y:y_end, x:x_end]

            # 如果块大小不足window_size，则需要进行填充
            if patch.shape[1] != patch_height or patch.shape[2] != patch_width:
                patch_padded = torch.zeros((image.shape[0], patch_height, patch_width), dtype=patch.dtype)
                patch_padded[:, :patch.shape[1], :patch.shape[2]] = patch
            else:
                patch_padded = patch

            # 模型预测
            patch_padded = patch_padded.unsqueeze(0).to(device)  # 增加批次维度 [1, C, H, W]
            with torch.no_grad():
                patch_pred = model(patch_padded)
                patch_pred = F.softmax(patch_pred.squeeze(0), dim=0)  # 进行Softmax归一化

            # 将预测结果放入全图的对应位置
            prediction_full[:, y:y_end, x:x_end] += patch_pred.cpu().numpy()[:, :y_end-y, :x_end-x]

    # 最后对每个像素点取argmax，得到最终的类别预测
    prediction_full = np.argmax(prediction_full, axis=0)

    return prediction_full

if __name__ == "__main__":
    # 是否使用GPU
    Cuda = True
    num_classes = 2
    predict_type = "Result"  # ConfidenceInterval or Result
    model_path = r"checkpoints/2024-10-13-17-32-25/model_state_dict_loss0.1429_epoch18.pth"

    input_shape = [512, 512]
    output_folder = r"data/output"
    data_dir = r"/mnt/ImarsData/ljs/project/pineapple_lacks_water/predict_whole/173D/F-3/DJI_202409230912_007_2024-09-17-173D"  
    # 图片所在的文件夹路径
    output_folder_name = os.path.basename(data_dir)
    output_folder = os.path.join(output_folder, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

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

    # 获取文件夹中的所有图片
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.tif', '.png', '.jpg', '.JPG'))]
    num_predict = len(image_files)

    print("device:", device, "num_predict:", num_predict)
    print("===============================================================================")

    image_transform = get_transform(input_shape, IsResize=False, IsTotensor=True, IsNormalize=True)

    print("start predicting")

    model.eval()
    for name_image in tqdm(image_files):
        image_path = os.path.join(data_dir, name_image)

        if name_image.endswith(".tif"):
            im_data, im_Geotrans, im_proj, cols, rows = read_tif(image_path)
        image = Image.open(image_path)
        name = os.path.basename(image_path)

        if image_transform is not None:
            image = image_transform(image)
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0, 1]))  # [C, H, W]

        image = image.to(device).float()

        # 使用滑动窗口进行预测
        prediction_full = sliding_window_predict(model, image, device, window_size=input_shape, step_size=256, num_classes=num_classes)

        if not os.path.exists(os.path.join(output_folder, name)):
            if name_image.endswith(".tif"):
                write_tif(os.path.join(output_folder, name), prediction_full, im_Geotrans, im_proj)
            else:
                cv2.imwrite(os.path.join(output_folder, name), prediction_full)

                # 可视化结果（归一化到0-255）
                vis_output_path = os.path.join(output_folder, name.split(".")[0] + '_visualization.png')

                prediction_min = prediction_full.min()
                prediction_max = prediction_full.max()

                if prediction_max != prediction_min:
                    prediction_normalized = ((prediction_full - prediction_min) / (prediction_max - prediction_min)) * 255
                else:
                    prediction_normalized = prediction_full

                prediction_normalized = prediction_normalized.astype('uint8')
                cv2.imwrite(vis_output_path, prediction_normalized)

    print("finish predicting successfully")
    print("===============================================================================")
