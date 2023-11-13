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

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def test(args):
    exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    work_dir = os.path.join(exp_name, 'Data')
    pair_list = list(open(os.path.join(work_dir, 'test_shift.txt')))  # list可以直接将打开的txt文件读取为一个列表
    result_name = "exp_result_Oneline-FastDLT_defocus06"
    result_files = os.path.join(exp_name, result_name)  # 结果文件的路径
    if not os.path.exists(result_files):  # 若结果文件路径不存在，则创建同名文件路径
        os.makedirs(result_files)

    result_txt = "result_ours_exp.txt"
    res_txt = os.path.join(result_files, result_txt)  # 打开一个txt文件
    f = open(res_txt, "w")

    net = build_model(args.model_name, pretrained=args.pretrained)  # 加载网络
    if args.finetune == True:
        model_path = os.path.join(exp_name, 'models/real_models/resnet34_iter_6000.pth')
        print(model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.state_dict().items():
            namekey = k[7:]  # remove `module.`
            new_state_dict[namekey] = v
        # load params
        net = build_model(args.model_name)
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

    print("start testing")
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
        # cv2.imshow("warp", pred_full)
        # cv2.imwrite("E:\\Postgraduate\\ImageAlign\\code\\DeepHomography-master\\testresult_s7\\{}.jpg".format(i),pred_full)
        # cv2.waitKey()
        # imagemerge = 0.5 * pred_full + 0.5 * print_img_2_d
        # cv2.imshow("merge",imagemerge/255.0)
        # cv2.waitKey()
        #fusion result
        img1 = print_img_2_d
        img2 = pred_full
        fusion = np.zeros((360, 640, 3), np.uint8)
        fusion[..., 0] = img2[..., 0]
        fusion[..., 1] = img1[..., 1] * 0.5 + img2[..., 1] * 0.5
        fusion[..., 2] = img1[..., 2]
        cv2.imwrite("E:\\Postgraduate\\ImageAlign\\code\\DeepHomography-master\\experiment\\testresult_6000_pre\\{}.jpg".format(i),
                    pred_full)

        pred_full = cv2.cvtColor(pred_full, cv2.COLOR_BGR2RGB)
        print_img_1_d = cv2.cvtColor(print_img_1_d, cv2.COLOR_BGR2RGB)
        print_img_2_d = cv2.cvtColor(print_img_2_d, cv2.COLOR_BGR2RGB)

        input_list = [print_img_1_d, print_img_2_d]
        output_list = [pred_full, print_img_2_d]
        # create_gif(input_list, os.path.join(result_files, name + "_input_[" + result_name + "].gif"))
        # create_gif(output_list, os.path.join(result_files, name + "_output_[" + result_name + "].gif"))

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=0, help='Number of splits')
    parser.add_argument('--cpus', type=int, default=1, help='Number of cpus')

    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=360)
    parser.add_argument('--patch_size_h', type=int, default=315)
    parser.add_argument('--patch_size_w', type=int, default=560)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-9, help='learning rate')

    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained waights?')
    parser.add_argument('--finetune', type=bool, default=True, help='Use pretrained waights?')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)