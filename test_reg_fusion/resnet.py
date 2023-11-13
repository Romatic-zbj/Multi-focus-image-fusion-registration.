import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch, imageio
from utils import transform, DLT_solve
import torch.nn.functional as F
#损失函数均方误差
criterion_l2 = nn.MSELoss(reduce=True, size_average=True)
#三元组损失，相关介绍可以参考 https://zhuanlan.zhihu.com/p/171627918
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1, reduce=False,size_average=False)

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

'''
创建一个gif图片
'''
def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def getPatchFromFullimg(patch_size_h, patch_size_w, patchIndices, batch_indices_tensor, img_full):
    '''
    随机从整幅图像中得到一个图像块
    patch_size_h:图像块高度
    patch_size_w：图像块宽度
    patchIndices：图像块下标
    batch_indices_tensor：
    image_full：整块图像
    '''
    num_batch, num_channels, height, width = img_full.size()
    # 计算输入图像的大小
    warped_images_flat = img_full.reshape(-1)#将图片reshape成为一维
    patch_indices_flat = patchIndices.reshape(-1)#将图片reshape成为一维

    pixel_indices = patch_indices_flat.long() + batch_indices_tensor
    mask_patch = torch.gather(warped_images_flat, 0, pixel_indices)#在相应的像素索引处取元素
    mask_patch = mask_patch.reshape([num_batch, 1, patch_size_h, patch_size_w])#reshape回去

    return mask_patch


def normMask(mask, strenth = 0.5):
    """
    :return: to attention more region
    将掩码进行归一化，用于关注某一些区域

    """
    batch_size, c_m, c_h, c_w = mask.size()
    max_value = mask.reshape(batch_size, -1).max(1)[0]# max会返回两个值，一个是最大值，另一个是最大值的索引
    max_value = max_value.reshape(batch_size, 1, 1, 1)# max_value reshape 成batchsize行，每行一个值
    mask = mask/(max_value*strenth)
    mask = torch.clamp(mask, 0, 1)

    return mask

# 定义一个3*3的卷积运算
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def extract_patches(x, kernel=3, stride=1):
    if kernel != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    return all_patches

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)#卷积
        self.bn1 = nn.BatchNorm2d(planes)#BN层
        self.relu = nn.ReLU(inplace=True)#激活层
        self.conv2 = conv3x3(planes, planes)#卷积
        self.bn2 = nn.BatchNorm2d(planes)#BN层
        self.downsample = downsample#下采样
        self.stride = stride#移动步幅

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# define and forward ( Because of the load is unbalanced when use torch.nn.DataParallel, we define warp in forward)
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)#1000分类，它在做什么？

        self.ShareFeature = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.genMask = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),#卷积层
                nn.BatchNorm2d(planes * block.expansion),#BN层
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # forward ( Because of the load is unbalanced when use torch.nn.DataParallel, we define warp in forward)
    def forward(self, org_imges, input_tesnors, h4p, patch_indices):

        batch_size, _, img_h, img_w = org_imges.size()# 获取原始图像shape
        _, _, patch_size_h, patch_size_w = input_tesnors.size() #获取输入tensor的shape

        y_t = torch.arange(0, batch_size * img_w * img_h,
                           img_w * img_h)
        batch_indices_tensor = y_t.unsqueeze(1).expand(y_t.shape[0], patch_size_h * patch_size_w).reshape(-1)

        M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()
            batch_indices_tensor = batch_indices_tensor.cuda()
            # org_imges=org_imges.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, M_tensor.shape[-2], M_tensor.shape[-1])
        # Inverse of M
        M_tensor_inv = torch.inverse(M_tensor)# 计算矩阵的逆
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, M_tensor_inv.shape[-2],
                                                      M_tensor_inv.shape[-1])# unqueeze表示扩张维度， expand表示改变tensor的形状

        mask_I1_full = self.genMask(org_imges[:, :1, ...])#其中的冒号表示对于后续维度进行省略
        mask_I2_full = self.genMask(org_imges[:, 1:, ...])#对后续维度进行省略

        mask_I1 = getPatchFromFullimg(patch_size_h, patch_size_w, patch_indices, batch_indices_tensor, mask_I1_full)
        mask_I2 = getPatchFromFullimg(patch_size_h, patch_size_w, patch_indices, batch_indices_tensor, mask_I2_full)

        mask_I1 = normMask(mask_I1)#归一化掩码

        mask_I2 = normMask(mask_I2)#归一化

        patch_1 = self.ShareFeature(input_tesnors[:, :1, ...])# 输入进去的tensor
        patch_2 = self.ShareFeature(input_tesnors[:, 1:, ...])

        patch_1_res = torch.mul(patch_1, mask_I1)
        patch_2_res = torch.mul(patch_2, mask_I2)
        x = torch.cat((patch_1_res, patch_2_res), dim=1)#现在是将图像输入到ResNet中
        #*********************************************************

        #***************************************

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)   #全连接层
        
        H_mat = DLT_solve(h4p, x).squeeze(1)#这里是直接进行仿射矩阵的计算

        pred_I2 = transform(patch_size_h, patch_size_w, M_tile_inv, H_mat, M_tile,
                            org_imges[:, :1, ...], patch_indices, batch_indices_tensor)
        pred_Mask = transform(patch_size_h, patch_size_w, M_tile_inv, H_mat, M_tile,
                            mask_I1_full, patch_indices, batch_indices_tensor)

        pred_Mask = normMask(pred_Mask)
 
        mask_ap = torch.mul(mask_I2, pred_Mask)

        # step1 freeze the mask_ap use "mask_ap = torch.ones_like(mask_ap)" ,thus gradient do not update and mask=1
        # step2 delete this line ("mask_ap = torch.ones_like(mask_ap)")  to update gradient of genMask
        # ######
        # mask_ap = torch.ones_like(mask_ap)
        # ######

        sum_value = torch.sum(mask_ap)
        pred_I2_CnnFeature = self.ShareFeature(pred_I2)
        patch_2=torch.unsqueeze(patch_2,dim=4)
        pred_I2_CnnFeature = torch.unsqueeze(pred_I2_CnnFeature, dim=4)
        patch_1 = torch.unsqueeze(patch_1, dim=4)
        feature_loss_mat = triplet_loss(patch_2, pred_I2_CnnFeature, patch_1)
        # print(patch_2.size())#16,1,315,560
        # print(pred_I2_CnnFeature.size())#16,1,315,560
        # print(patch_1.size())#16,1,315,560
        # print(feature_loss_mat.size())#16,1,315
        # print(mask_ap.size())#16,1,315,560
        feature_loss = torch.sum(torch.mul(feature_loss_mat, mask_ap)) / sum_value
        feature_loss = torch.unsqueeze(feature_loss, 0)

        pred_I2_d = pred_I2[:1, ...]
        patch_2_res_d = patch_2_res[:1, ...]
        pred_I2_CnnFeature_d = pred_I2_CnnFeature[:1, ...]
        mask_ap_d = mask_ap[:1, ...]
        feature_loss_mat_d = feature_loss_mat[:1, ...]

        out_dict = {}
        out_dict.update(feature_loss=feature_loss, pred_I2_d=pred_I2_d, x=x, H_mat=H_mat, patch_2_res_d=patch_2_res_d,
                        pred_I2_CnnFeature_d=pred_I2_CnnFeature_d, mask_ap_d=mask_ap_d.squeeze(1), feature_loss_mat_d=feature_loss_mat_d)
        
        return out_dict


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

