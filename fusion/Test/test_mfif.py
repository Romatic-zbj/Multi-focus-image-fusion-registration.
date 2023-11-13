import sys

sys.path.append("..")

import argparse
import pathlib
import warnings
import statistics
import time

import os
import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
import torchvision
from torch import Tensor
from tqdm import tqdm

from dataloader.test_mifi_data import FuseTestYcrbr
from models.fusion_net import FusionNet
EPS = 1e-8

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='Fuse Net eval process')
    # dataset
    parser.add_argument('--imgL',      default='./img1', type=pathlib.Path)
    parser.add_argument('--imgR',      default='./img2', type=pathlib.Path)
    # checkpoint
    parser.add_argument('--ckpt', default='../checkpoints/best_0400.pth', help='weight checkpoint', type=pathlib.Path) # weight/default.pth
    parser.add_argument('--dst', default='', help='fuse image save folder', type=pathlib.Path)

    parser.add_argument('--dim', default=64, type=int, help='AFuse feather dim')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")

    args = parser.parse_args()
    return args
def main(args):

    # cuda = args.cuda
    # if cuda and torch.cuda.is_available():
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # else:
    #     raise Exception("No GPU found...")
    # torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("===> Loading datasets")
    data = FuseTestYcrbr(args.imgL, args.imgR)
    test_data_loader = torch.utils.data.DataLoader(data, 1, True, pin_memory=True)

    print("===> Building model")
    net = FusionNet(nfeats=args.dim).to(device)

    print("===> loading trained model '{}'".format(args.ckpt))
    model_state_dict = torch.load(args.ckpt,map_location=torch.device('cpu'))
    net.load_state_dict(model_state_dict)

    print("===> Starting Testing")
    test(net, test_data_loader, args.dst, device)
def test(net, test_data_loader, dst, device):
    net.eval()

    fus_time = []
    tqdm_loader = tqdm(test_data_loader, disable=True)
    for img, (ir_path, vi_path) in tqdm_loader:

        name, ext = os.path.splitext(os.path.basename(ir_path[0]))
        pre, order, nex = name.split('-')
        file_name = name + ext
        imgs = torch.cat(img,dim=0)
        print(imgs.size())

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
        start = time.time()
        with torch.no_grad():
            fused_img_y  = net(img1_y, img2_y)
        fused_img_y = (fused_img_y + 1) / 2 #归一化到0-1区间
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        end = time.time()
        fus_time.append(end - start)


        fuse_out = torch.cat((fused_img_y,fused_img_cr, fused_img_cb),dim=1)#Ycrbr格式
        # print(fuse_out.size())
        # fused_img = (fuse_out + 1) * 127.5
        fused_img = fuse_out.squeeze(0)
        fused_img = fused_img.cpu().numpy()
        fused_img = np.transpose(fused_img, (1, 2, 0))*255.0
        fused_img = fused_img.astype(np.uint8)
        fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)
        path = "./test_result_Ly_s0606_/fus_300/"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = "color_lytro_"+ order + ".jpg"
        cv2.imwrite(path + file_name,fused_img)

        # TODO: save fused images
        # imsave(fused_img, dst / 'fused_reg' / file_name)
        # cv2.imshow("fused",fused_img)
        # cv2.waitKey()
        # imsave(img[0], dst / 'ir' / file_name)
        # imsave(img[1], dst / 'vi' / file_name)

    # statistics time record
    print(fus_time)
    fuse_mean = statistics.mean(fus_time[:])
    print('fuse time (average): {:.4f}'.format(fuse_mean))
    print('fps (equivalence): {:.4f}'.format(1. / fuse_mean))

    pass

def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts)*255.

        cv2.imwrite(str(p), im_cv)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = hyper_args()
    main(args)
