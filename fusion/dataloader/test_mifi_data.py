import pathlib

import cv2
import numpy as np
import kornia.utils
import torch.utils.data
import torchvision.transforms.functional
from PIL import Image


class FuseTestData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path):
        super(FuseTestData, self).__init__()
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]
        #确保这两个需要融合的图片名称一致
        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        irB, irG, irR = self.imread(path=ir_path)
        viB, viG, viR = self.imread(path=vi_path)

        return (irB, irG, irR, viB, viG, viR), (str(ir_path), str(vi_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path):
        img = cv2.imread(str(path))
        imgB, imgG, imgR = cv2.split(img)

        imB_ts = kornia.utils.image_to_tensor(imgB / 255.).type(torch.FloatTensor)
        imG_ts = kornia.utils.image_to_tensor(imgG / 255.).type(torch.FloatTensor)
        imR_ts = kornia.utils.image_to_tensor(imgR / 255.).type(torch.FloatTensor)
        return imB_ts, imG_ts, imR_ts
class FuseTestYcrbr(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path):
        super(FuseTestYcrbr, self).__init__()
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]
        #确保这两个需要融合的图片名称一致
        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        img1 = self.imread(path=ir_path)
        # print(img1.size())
        img2 = self.imread(path=vi_path)

        return (img1, img2), (str(ir_path), str(vi_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = kornia.utils.image_to_tensor(img / 255.).type(torch.FloatTensor)
        # print(img.size())
        return img
