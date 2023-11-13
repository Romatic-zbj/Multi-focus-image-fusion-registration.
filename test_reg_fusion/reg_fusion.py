#coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from torch_homography_model import build_model
from dataset_no_npy import *
from utils import transformer as trans
import os
import numpy as np
import cv2
import kornia
from fusion_models.fusion_net import FusionNet
import pathlib

#配准
def reg(args):
    exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    print(exp_name)
    exp_name = os.path.join(exp_name,'test_reg_fusion')
    model_path = os.path.join(exp_name,'models','resnet34_iter_6000.pth')
    state_dict = torch.load(model_path,map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.state_dict().items():
        namekey = k[7:]  # remove `module.`
        new_state_dict[namekey] = v
    # load params
    net = build_model("resnet34")
    model_dict = net.state_dict()
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
    model_dict.update(new_state_dict)
    net.load_state_dict(model_dict)

    if torch.cuda.is_available():
        net = net.cuda()
    M_tensor = torch.tensor([[args.img_w / 2.0, 0., args.img_w / 2.0],
                             [0., args.img_h / 2.0, args.img_h / 2.0],
                             [0., 0., 1.]])
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()
    M_tile = M_tensor.unsqueeze(0).expand(1, M_tensor.shape[-2], M_tensor.shape[-1])
    # Inverse of M
    M_tensor_inv = torch.inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(1, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])

    test_data = TestDataset(data_path=exp_name, patch_w=args.patch_size_w, patch_h=args.patch_size_h, rho=16,
                            WIDTH=args.img_w, HEIGHT=args.img_h)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0, shuffle=False, drop_last=True)
    net.eval()
    for i, batch_value in enumerate(test_loader):
        org_imges = batch_value[0].float()
        input_tesnors = batch_value[1].float()
        patch_indices = batch_value[2].float()
        h4p = batch_value[3].float()
        print_img_1 = batch_value[4]
        print_img_2 = batch_value[5]

        print_img_1_d = print_img_1.cpu().detach().numpy()[0, ...]
        print_img_2_d = print_img_2.cpu().detach().numpy()[0, ...]
        print_img_1_d = np.transpose(print_img_1_d, [1, 2, 0])  # 就是将通道数调整到后面
        print_img_2_d = np.transpose(print_img_2_d, [1, 2, 0]) #已经放在cpu上了，

        if torch.cuda.is_available():
            input_tesnors = input_tesnors.cuda()
            patch_indices = patch_indices.cuda()
            h4p = h4p.cuda()
            print_img_1 = print_img_1.cuda()
            org_imges = org_imges.cuda()

        batch_out = net(org_imges, input_tesnors, h4p, patch_indices)
        H_mat = batch_out['H_mat']#仿射矩阵被计算出来

        output_size = (args.img_h, args.img_w)

        H_point = H_mat.squeeze(0)
        H_point = H_point.cpu().detach().numpy()
        H_point = np.linalg.inv(H_point)  # 求逆矩阵
        H_point = (1.0 / H_point.item(8)) * H_point

        # print(H_point)
        name = "0" * (8 - len(str(i))) + str(i)

        H_mat = torch.matmul(torch.matmul(M_tile_inv, H_mat), M_tile)
        # print(H_mat)
        pred_full, _ = trans(print_img_1, H_mat, output_size)  # pred_full = warped imgA
        pred_full = pred_full.cpu().detach().numpy()[0, ...]
        pred_full = pred_full.astype(np.uint8)

    return print_img_2_d, pred_full #BGR格式的numpy,即输出为B,A‘
#裁剪
def auto_crop(img,imgB):
    _,thresh = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1,255,cv2.THRESH_BINARY)
    left,right,upper,lower = [-1,-1,-1,-1]
    black_pixel_num_threshold_h = img.shape[1]//10#用于上下裁剪的阈值
    black_pixel_num_threshold_w = img.shape[0]//10#用于左右裁剪的阈值
    for y in range(thresh.shape[0]):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold_h:
            upper = y
            break

    for y in range(thresh.shape[0] - 1, 0, -1):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold_h:
            lower = y
            break
    for x in range(thresh.shape[1]):
        if len(np.where(thresh[:, x] == 0)[0]) < black_pixel_num_threshold_w:
            left = x
            break

    for x in range(thresh.shape[1] - 1, 0, -1):
        if len(np.where(thresh[:, x] == 0)[0]) < black_pixel_num_threshold_w:
            right = x
            break
    return img[upper:lower,left:right],imgB[upper:lower,left:right]
#自定义裁剪，指定裁剪起始点，宽高

def custom_crop(img, imgB, start_point_x, start_point_y, height, width):

    cropped_imgA = img[start_point_y:start_point_y + height, start_point_x:start_point_x + width]
    cropped_imgB = imgB[start_point_y:start_point_y + height, start_point_x:start_point_x + width]
    return cropped_imgA,cropped_imgB

#融合
def fusion(args,imgA,imgB):
    EPS=EPS = 1e-8
    # 先进行裁剪(这里使用了自定义裁剪，也可以使用自动裁剪)
    img1, img2 = custom_crop(imgA, imgB, (32, 9), 340, 400)
    # 确保两张图象尺寸一致
    assert img1.shape == img2.shape, 'make sure two pictures have same size'
    # 再进行融合
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = FusionNet(nfeats=args.dim).to(device)
    model_state_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))

    img1_tensor = BGR2YCbCr(img1) # 将裁剪后的numpy转化为tensor
    img2_tensor = BGR2YCbCr(img2) # 将裁剪后的numpy转化为tensor
    imgs = torch.cat((img1_tensor,img2_tensor), dim=0)
    img1_y = imgs[0:1, 0:1, :, :]
    img2_y = imgs[1:2, 0:1, :, :]
    img_cr = imgs[:, 1:2, :, :]
    # print(img_cr.size())
    img_cb = imgs[:, 2:3, :, :]
    w_cr = (torch.abs(img_cr) + EPS) / torch.sum(torch.abs(img_cr) + EPS, dim=0)
    w_cb = (torch.abs(img_cb) + EPS) / torch.sum(torch.abs(img_cb) + EPS, dim=0)
    fused_img_cr = torch.sum(w_cr * img_cr, dim=0, keepdim=True).clamp(-1, 1)
    fused_img_cb = torch.sum(w_cb * img_cb, dim=0, keepdim=True).clamp(-1, 1)
    # Fusion
    torch.cuda.synchronize() if str(device) == 'cuda' else None
    with torch.no_grad():
        fused_img_y = net(img1_y, img2_y)
    fused_img_y = (fused_img_y + 1) / 2  # 归一化到0-1区间
    torch.cuda.synchronize() if str(device) == 'cuda' else None


    fuse_out = torch.cat((fused_img_y, fused_img_cr, fused_img_cb), dim=1)  # Ycrbr格式
    # print(fuse_out.size())
    # fused_img = (fuse_out + 1) * 127.5
    fused_img = fuse_out.squeeze(0)
    fused_img = fused_img.cpu().numpy()
    fused_img = np.transpose(fused_img, (1, 2, 0)) * 255.0
    fused_img = fused_img.astype(np.uint8)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)
    return fused_img



#图像格式转换
def BGR2YCbCr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img = kornia.utils.image_to_tensor(img / 255.).type(torch.FloatTensor)
    # print(img.size())
    return img




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=0, help='Number of splits')
    parser.add_argument('--cpus', type=int, default=1, help='Number of cpus')
    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=360)
    parser.add_argument('--patch_size_h', type=int, default=315)
    parser.add_argument('--patch_size_w', type=int, default=560)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--ckpt', default='./fuse_checkpoints/best.pth', help='weight checkpoint', type=pathlib.Path)
    parser.add_argument('--dim',default=64, type=int, help='AFuse feather dim')

    args = parser.parse_args()
    imgB, imgA = reg(args)
    result = fusion(args,imgA,imgB)
    cv2.imwrite("./result.jpg",result)
