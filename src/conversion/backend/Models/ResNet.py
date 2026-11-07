"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100, init_channels: int = 3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(init_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  #1relu
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1) #3*3
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2) #3*4
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2) #3*6
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2) #3*3
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_layer = nn.Linear(512 * block.expansion, num_classes)
        self.initialize_weights_kaiming()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))  #for bottleneck 3*num_blocks relu
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def initialize_weights_kaiming(model):
        for m in model.modules():
            # Conv2d 初始化（你的实验 bias=True 也支持）
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # BatchNorm2d
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # Linear
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.last_layer(output)
        return output

class ResNet4Cifar(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_layer = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.last_layer(output)
        return output

def resnet18(num_classes=10, in_channels=3, **kargs):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, init_channels=in_channels)

def resnet20(num_classes=10, **kargs):
    """ return a ResNet 20 object
    """
    return ResNet4Cifar(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet34(num_classes=10, **kargs):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10, in_channels=3, **kargs):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, init_channels=in_channels)

def resnet101(num_classes=10, **kargs):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=num_classes)

def resnet152(num_classes=10, **kargs):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3],num_classes=num_classes)

if __name__=='__main__':
    if __name__=='__main__':
        model = resnet18(10)
        model.eval()
        import torch
        with torch.no_grad():
            conv_num = 0.
            linear_num = 0.
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    linear_num += module.weight.numel()
                elif isinstance(module, torch.nn.Conv2d):
                    conv_num += module.weight.numel()
            print(f"conv {conv_num}, linear {linear_num}, ratio {linear_num/conv_num}")
