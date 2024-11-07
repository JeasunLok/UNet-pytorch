import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch 
import torch.distributed as dist
import random
import rasterio

# 读取tif
def read_tif(path):
    with rasterio.open(path) as dataset:
        im_data = dataset.read()  # 读取数据
        if len(im_data) == 2:  # 处理单波段情况
            im_data = im_data[np.newaxis, :, :]  # 添加波段维度
        im_data = np.transpose(im_data, [1, 2, 0])  # 转置为 (height, width, channels)
        im_proj = dataset.crs  # 读取投影
        im_geotrans = dataset.transform  # 读取仿射变换
        cols, rows = dataset.width, dataset.height
    return im_data, im_geotrans, im_proj, cols, rows


# 写出tif
def write_tif(newpath, im_data, im_geotrans, im_proj):
    if len(im_data) == 2:  # 处理二维数据的情况
        im_data = im_data[np.newaxis, :, :]  # 添加一个波段维度
    bands = im_data.shape[0]
    height = im_data.shape[1]
    width = im_data.shape[2]
    datatype = im_data.dtype  # 获取数据类型

    with rasterio.open(newpath, 'w', driver='GTiff', height=height, 
                       width=width, count=bands, 
                       dtype=datatype, crs=im_proj, transform=im_geotrans) as new_dataset:
        for i in range(bands):
            new_dataset.write(im_data[i, :, :], i + 1)

def init_ddp(local_rank):
    '''
    有了这一句之后,在转换device的时候直接使用 a=a.cuda()即可,否则要用a=a.cuda(local_rank)
    '''
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

def reduce_tensor(tensor: torch.Tensor):
    '''
    对多个进程计算的多个 tensor 类型的 输出值取平均操作
    '''
    rt = tensor.clone()  # tensor(9.1429, device='cuda:1')
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt

def get_ddp_generator(seed=3407):
    '''
    对每个进程使用不同的随机种子，增强训练的随机性
    '''
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# a class for calculating the average of the accuracy and the loss
#-------------------------------------------------------------------------------
class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.average = 0 
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.count += n
    self.average = self.sum / self.count
#-------------------------------------------------------------------------------

def transform(IsResize,Resize_size,IsTotensor,IsNormalize,Norm_mean,Norm_std,IsRandomGrayscale,IsColorJitter,
              brightness,contrast,hue,saturation,IsCentercrop,Centercrop_size,IsRandomCrop,RandomCrop_size,
              IsRandomResizedCrop,RandomResizedCrop_size,Grayscale_rate,IsRandomHorizontalFlip,HorizontalFlip_rate,
              IsRandomVerticalFlip,VerticalFlip_rate,IsRandomRotation,degrees):

  transform_list = []

    #-----------------------------------------------<旋转图像>-----------------------------------------------------------#
  if IsRandomRotation:
    transform_list.append(transforms.RandomRotation(degrees))
  if IsRandomHorizontalFlip:
    transform_list.append(transforms.RandomHorizontalFlip(HorizontalFlip_rate))
  if IsRandomVerticalFlip:
    transform_list.append(transforms.RandomHorizontalFlip(VerticalFlip_rate))

    #-----------------------------------------------<图像颜色>-----------------------------------------------------------#
  if IsColorJitter:
    transform_list.append(transforms.ColorJitter(brightness,contrast,saturation,hue))
  if IsRandomGrayscale:
    transform_list.append(transforms.RandomGrayscale(Grayscale_rate))

    #---------------------------------------------<缩放或者裁剪>----------------------------------------------------------#
  if IsResize:
    transform_list.append(transforms.Resize(Resize_size))
  if IsCentercrop:
    transform_list.append(transforms.CenterCrop(Centercrop_size))
  if IsRandomCrop:
    transform_list.append(transforms.RandomCrop(RandomCrop_size))
  if IsRandomResizedCrop:
    transform_list.append(transforms.RandomResizedCrop(RandomResizedCrop_size))

    #---------------------------------------------<tensor化和归一化>------------------------------------------------------#
  if IsTotensor:
    transform_list.append(transforms.ToTensor())
  if IsNormalize:
    transform_list.append(transforms.Normalize(Norm_mean,Norm_std))

    # 您可以更改数据增强的顺序，但是数据增强的顺序可能会影响最终数据的质量，因此除非您十分明白您在做什么,否则,请保持默认顺序
  # transforms_order=[Resize_transform,Rotation,Color,Tensor,Normalize]
  return transforms.Compose(transform_list)


def get_transform(size=[200, 200], mean=[0, 0, 0], std=[1, 1, 1], IsResize=False, IsCentercrop=False, IsRandomCrop=False, IsRandomResizedCrop=False, IsTotensor=False, IsNormalize=False, IsRandomGrayscale=False, IsColorJitter=False, IsRandomVerticalFlip=False, IsRandomHorizontalFlip=False, IsRandomRotation=False):
  diy_transform = transform(
      IsResize=IsResize, #是否缩放图像
      Resize_size=size, #缩放后的图像大小 如（512,512）->（256,192）
      IsCentercrop=IsCentercrop,#是否进行中心裁剪
      Centercrop_size=size,#中心裁剪后的图像大小
      IsRandomCrop=IsRandomCrop,#是否进行随机裁剪
      RandomCrop_size=size,#随机裁剪后的图像大小
      IsRandomResizedCrop=IsRandomResizedCrop,#是否随机区域进行裁剪
      RandomResizedCrop_size=size,#随机裁剪后的图像大小
      IsTotensor=IsTotensor, #是否将PIL和numpy格式的图片的数值范围从[0,255]->[0,1],且将图像形状从[H,W,C]->[C,H,W]
      IsNormalize=IsNormalize, #是否对图像进行归一化操作,即使用图像的均值和方差将图像的数值范围从[0,1]->[-1,1]
      Norm_mean=mean,#图像的均值，用于图像归一化，建议使用自己通过计算得到的图像的均值
      Norm_std=std,#图像的方差，用于图像归一化，建议使用自己通过计算得到的图像的方差
      IsRandomGrayscale=IsRandomGrayscale,#是否随机将彩色图像转化为灰度图像
      Grayscale_rate=0.5,#每张图像变成灰度图像的概率，设置为1的话等同于transforms.Grayscale()
      IsColorJitter=IsColorJitter,#是否随机改变图像的亮度、对比度、色调和饱和度
      brightness=0.5,#每个图像被随机改变亮度的概率
      contrast=0.5,#每个图像被随机改变对比度的概率
      hue=0.5,#每个图像被随机改变色调的概率
      saturation=0.5,#每个图像被随机改变饱和度的概率
      IsRandomVerticalFlip=IsRandomVerticalFlip,#是否垂直翻转图像
      VerticalFlip_rate=0.5,#每个图像被垂直翻转图像的概率
      IsRandomHorizontalFlip=IsRandomHorizontalFlip,#是否水平翻转图像
      HorizontalFlip_rate=0.5,#每个图像被水平翻转图像的概率
      IsRandomRotation=IsRandomRotation,#是是随机旋转图像
      degrees=10,#每个图像被旋转角度的范围 如degrees=10 则图像将随机旋转一个(-10,10)之间的角度
  )
  return diy_transform

def compute_mIoU(CM, ignore_index=-1):
    np.seterr(divide="ignore", invalid="ignore")
    if ignore_index >= 0:
      CM = np.delete(CM, ignore_index, axis=0)
      CM = np.delete(CM, ignore_index, axis=1)
    intersection = np.diag(CM)
    ground_truth_set = CM.sum(axis=1)
    predicted_set = CM.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    mIoU = np.mean(IoU)
    return mIoU

def compute_acc(CM, ignore_index=-1):
    np.seterr(divide="ignore", invalid="ignore")
    if ignore_index >= 0:
      CM = np.delete(CM, ignore_index, axis=0)
      CM = np.delete(CM, ignore_index, axis=1)
    TP = np.sum(np.diag(CM))
    Sum = np.sum(CM)
    acc = TP/Sum
    return acc

def compute_metrics(CM, ignore_index=-1):
    np.seterr(divide="ignore", invalid="ignore")
    if ignore_index >= 0:
      CM = np.delete(CM, ignore_index, axis=0)
      CM = np.delete(CM, ignore_index, axis=1)
    num_classes = CM.shape[0]
    GT_array = np.sum(CM, axis=0)
    TP_array = np.diag(CM)
    Recall_array = np.array([])
    Precision_array = np.array([])
    F1_array = np.array([])

    for i in range(num_classes):
        TP = TP_array[i]
        FP = np.sum(CM[i,:])-TP
        FN = np.sum(CM[:,i])-TP
        TN = np.sum(CM)-TP-FP-FN
        # print(i, TP, FP, FN, TN)
        Recall = TP/(TP+FN)
        Precision = TP/(TP+FP)
        F1 = 2*Recall*Precision/(Recall+Precision)
        Recall_array = np.append(Recall_array, Recall)
        Precision_array = np.append(Precision_array, Precision)
        F1_array = np.append(F1_array, F1)
    # print(Recall_array, Precision_array, F1_array)
    weighted_Recall = np.sum(GT_array*Recall_array)/np.sum(CM)
    weighted_Precision = np.sum(GT_array*Precision_array)/np.sum(CM)
    weighted_F1 = np.sum(GT_array*F1_array)/np.sum(CM)
    
    return weighted_Recall, weighted_Precision, weighted_F1

def draw_result_visualization(folder, epoch_result):
    # the change of loss
    np.savetxt(os.path.join(folder, "epoch.txt"), epoch_result, fmt="%.4f", delimiter=',', newline='\n')
    with plt.ioff():
      plt.figure()
      plt.plot(epoch_result[:][0], epoch_result[:][1])
      plt.title("the change of the loss")
      plt.xlabel("epoch")
      plt.ylabel("loss")
      plt.savefig(os.path.join(folder, "loss_change.png"))
      plt.figure()
      plt.plot(epoch_result[:][0], epoch_result[:][2])
      plt.title("the change of the accuracy")
      plt.xlabel("epoch")
      plt.ylabel("accuracy")
      plt.savefig(os.path.join(folder, "accuracy_change.png"))
      plt.figure()
      plt.plot(epoch_result[:][0], epoch_result[:][3])
      plt.title("the change of the MIoU")
      plt.xlabel("epoch")
      plt.ylabel("MIoU")
      plt.savefig(os.path.join(folder, "MIoU_change.png"))

def store_result(folder, Accuracy, mIoU, W_Recall, W_Precision, W_F1, CM, epoch, batch_size, learning_rate, weight_decay):
    with open(os.path.join(folder, "accuracy.txt"), 'w', encoding="utf-8") as f:
        f.write("Parameter settings:" + "\n")
        f.write("epoch : " + str(epoch) + "\n")
        f.write("batch_size : " + str(batch_size) + "\n")
        f.write("learning_rate : " + str(learning_rate) + "\n")
        f.write("weight_decay : " + str(weight_decay) + "\n")
        f.write("Model result:" + "\n")
        f.write("Accuracy : {:.4f}\n".format(Accuracy))
        f.write("mIoU : {:.4f}\n".format(mIoU))
        f.write("W-Recall : {:.3f}\n".format(W_Recall))
        f.write("W-Precision : {:.3f}\n".format(W_Precision))
        f.write("W-F1 : {:.3f}\n".format(W_F1))
        f.write("Confusion Matrix :\n")
        f.write("{}".format(CM))

