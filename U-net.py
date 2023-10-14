import torch
from torch.nn import functional as F
from torch import nn
import torchvision
from torchsummary import summary

pretrained_net_cod = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
resnet_cod_layers = list(pretrained_net_cod.children())[:-2]


# summary(pretrained_net_cod, (3, 512, 256), device='cpu')
# print(pretrained_net_cod)

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

class TransposeBlock(nn.Module):
    def __init__(self, num_channels, decrease_channels=False, increase_size=False):
        super().__init__()

        if increase_size:
            self.stride = 2
            self.output_padding = 1
        else:
            self.stride = 1
            self.output_padding = 0

        self.convT1 = nn.LazyConvTranspose2d(num_channels, 3, padding=1, bias=False,
                                             stride=self.stride, output_padding=self.output_padding)
        self.bn1 = nn.LazyBatchNorm2d()

        self.conv2 = nn.LazyConv2d(num_channels, 3, padding=1, bias=False, )
        self.bn2 = nn.LazyBatchNorm2d()

        if decrease_channels or increase_size:
            self.feedforward = nn.LazyConvTranspose2d(num_channels, 3, padding=1, bias=False,
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

def flatten_layers(model):
    flat_layers = []

    for layer in model:
        if isinstance(layer, nn.Sequential):
            flat_layers.extend([flatten_layers(list(layer.children()))])
        elif isinstance(layer, torchvision.models.resnet.BasicBlock) or isinstance(layer, TransposeBlock) or isinstance(layer, Top):
            flat_layers.extend([flatten_layers(list(layer.children()))])
        elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            flat_layers.append(layer)

    return flat_layers


resnet_weight = flatten_layers(list(pretrained_net_cod.children())[:-2])

class ResNet(nn.Module):
    def __init__(self, num_channels, num_classes=2):
        super().__init__()
        self.net = nn.Sequential()
        for i, ch in enumerate(num_channels):
            self.net.add_module(
                f'b{i + 1}',
                TransposeBlock(ch, decrease_channels=(i % 2 == 0) and (i != 0),
                               increase_size=(i % 2 == 0) and (i != 0)))
            self.net[-1].conv2.weight.data = resnet_weight[-(int(i/2)+1)][i % 2][0].weight
            self.net[-1].convT1.weight.data = resnet_weight[-(int(i/2)+1)][i % 2][1].weight
            if self.net[-1].feedforward:
                print(i)
                self.net[-1].feedforward.weight.data = resnet_weight[-(int(i/2)+1)][i % 2][-1][0].weight

        self.net.add_module('top',
                            nn.LazyConvTranspose2d(num_classes, 7, stride=4, padding=3, output_padding=3, bias=False))
        self.net[-1].top.weight.data = resnet_weight[0].weight
    def forward(self, X):
        return self.net(X)


resnet_dec = ResNet((512, 512, 256, 256, 128, 128, 64, 64), 2)
# resnet_dec = ResNet((256, 256, 128, 128, 64, 64, 32, 32), 2)

# net = nn.Sequential(*resnet_cod_layers)
# net.add_module('deco', resnet_dec)

resnet_dec_layers = flatten_layers(list(resnet_dec.children()))
resnet_cod_layers = flatten_layers(list(pretrained_net_cod.children())[:-2])

# resnet_cod_layers = list(filter(lambda elemento: not isinstance(elemento, nn.ReLU)
#                                                  and not isinstance(elemento, nn.MaxPool2d), resnet_cod_layers))
# resnet_dec_layers = list(filter(lambda elemento: not isinstance(elemento, nn.ReLU)
#                                                  and not isinstance(elemento, nn.MaxPool2d), resnet_dec_layers))

# summary(net, (3, 512, 512), device='cpu')
def transfer_weights(reverse_list_model1, list_model2):
    lenght = len(reverse_list_model1) if (list_model2[-1].out_channels == 3) else len(reverse_list_model1)

    for i in range(lenght):
        if (isinstance(reverse_list_model1[i], nn.Conv2d) and
                (isinstance(list_model2[i - 1], nn.Conv2d) or isinstance(list_model2[i - 1],
                                                                               nn.ConvTranspose2d))):
            inverted_weights = torch.flip(reverse_list_model1[i].weight, (2, 3))

            list_model2[i - 1].weight = nn.Parameter(inverted_weights)

    if lenght != len(reverse_list_model1):
        mean_weights = torch.flip(torch.unsqueeze(torch.mean(reverse_list_model1[-1].weight, dim=1), 1), (2, 3))
        inverted_weights = torch.cat((mean_weights, mean_weights), dim=1)
        list_model2[-1].weight = nn.Parameter(inverted_weights)


# transfer_weights(resnet_cod_layers[::-1], resnet_dec_layers)