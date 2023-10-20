import torch
from torch.nn import functional as F
from torch import nn
import torchvision
from torchsummary import summary
import numpy as np

pretrained_net_cod = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
resnet_cod_layers = list(pretrained_net_cod.children())[:-2]
path = 'C:\\Proxectos\\Custom_U-NET\\g.v2i.coco-segmentation'

def flatten_layers(model):
    flat_layers = []

    for layer in model:
        if isinstance(layer, nn.Sequential):
            flat_layers.extend([flatten_layers(list(layer.children()))])
        elif isinstance(layer, torchvision.models.resnet.BasicBlock): #or isinstance(layer, TransposeBlock) or isinstance(layer, Top):
            flat_layers.extend([flatten_layers(list(layer.children()))])
        elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            flat_layers.append(layer)

    return flat_layers


resnet_weight = flatten_layers(list(pretrained_net_cod.children())[:-2])

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

class TransposeBlock(nn.Module):
    def __init__(self, out_channels, decrease_channels=False, increase_size=False):
        super().__init__()

        if increase_size:
            self.stride = 2
            self.output_padding = 1
        else:
            self.stride = 1
            self.output_padding = 0

        in_channels = out_channels if not decrease_channels else int(out_channels * 2)

        self.convT1 = nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1, bias=False,
                                         stride=self.stride, output_padding=self.output_padding)
        self.bn1 = nn.LazyBatchNorm2d()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, )
        self.bn2 = nn.LazyBatchNorm2d()

        if decrease_channels or increase_size:
            self.feedforward = nn.ConvTranspose2d(in_channels, out_channels, 1, padding=0, bias=False,
                                                  stride=self.stride, output_padding=self.output_padding)
        else:
            self.feedforward = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.convT1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.feedforward:
            X = self.feedforward(X)
        Y += X
        return F.relu(Y)


class Top(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConvTranspose2d(num_classes, 7, stride=4, padding=3, output_padding=3, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.net(X)



class ResNet(nn.Module):
    def __init__(self, num_channels, num_classes=2):
        super().__init__()
        self.net = nn.Sequential()
        for i, ch in enumerate(num_channels):
            self.net.add_module(
                f'b{i + 1}',
                TransposeBlock(ch, decrease_channels=(i % 2 != 0) and (i != len(num_channels)-1),
                               increase_size=(i % 2 != 0) and (i != len(num_channels)-1)))
            # X = torch.rand(1, 512, 32, 32)
            # A = self.net(X)
            self.net[-1].convT1.weight.data.copy_(torch.flip(resnet_weight[-(int(i/2)+1)][1 - i % 2][0].weight, (2, 3)));
            if 1 - i % 2 or (i == len(num_channels)-1):
                self.net[-1].conv2.weight.data.copy_(torch.flip(resnet_weight[-(int(i/2)+1)][1 - i % 2][1].weight, (2, 3)));
            else:
                weights = resnet_weight[-(int(i / 2) + 1)][1 - i % 2][1].weight
                resize_weights = np.resize(weights.detach(), (int(weights.shape[0]/2), int(weights.shape[1]/2), weights.shape[2], weights.shape[3]))
                self.net[-1].conv2.weight.data.copy_(torch.flip(nn.Parameter(torch.Tensor(resize_weights)), (2, 3)));
            if self.net[-1].feedforward:
                self.net[-1].feedforward.weight.data.copy_(torch.flip(resnet_weight[-(int(i/2)+1)][1 - i % 2][-1][0].weight,(2, 3)));

        self.net.add_module('top',
                            nn.ConvTranspose2d(num_channels[-1], num_classes, 7, stride=4, padding=3, output_padding=3, bias=False))

        resize_weights = np.resize(resnet_weight[0].weight.detach(), (64, num_classes, 7, 7))
        # self.net[-1].weight.data = nn.Parameter(torch.flip(torch.tensor(resize_weights), (2, 3)))
        self.net[-1].weight.data.copy_(torch.flip(nn.Parameter(torch.tensor(resize_weights)), (2, 3)));
    def forward(self, X):
        return self.net(X)

def create_deco():
    input_shape = (512, 32, 32)
    resnet_dec = ResNet((512, 256, 256, 128, 128, 64, 64, 64), 2)
    summary(resnet_dec, input_shape, device='cpu')
    print('Input shape = '+str(input_shape))
    return resnet_dec

def create_unet():
    input_shape = (3, 512, 512)
    net = nn.Sequential(*resnet_cod_layers)
    resnet_dec = ResNet((512, 256, 256, 128, 128, 64, 64, 64), 2)
    net.add_module('deco', resnet_dec)
    print(summary(net, input_shape, device='cpu'))
    print('Input shape = '+str(input_shape))
    return net
