import torch
import MinkowskiEngine as ME
from model.autoencoder import InceptionResNet


def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))
        
    return torch.nn.Sequential(*layers)


class Predictor(torch.nn.Module):
    def __init__(self, channels=[1,16,32,64,32,8]):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[2]*2)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2]*2,
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[3]*2)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3]*2,
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.block2 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[4]*2)
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channels[4]*2,
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.conv_coord = ME.MinkowskiConvolution(
            in_channels=channels[5]*2,
            out_channels=channels[5]*2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block3 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[5]*2)
        
        self.conv4 = ME.MinkowskiConvolution(
            in_channels=channels[5]*2,
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # self.dp = ME.MinkowskiDropout()

    def forward(self, target, x_list, device):
        ## x_list strides/channels = [8/8, 8/32, 4/64, 2/32, 1/16]
        out0 = self.relu(self.conv0(x_list[4]))
        out1 = self.relu(self.down0(out0))
        # out1 = self.dp(out1)
        
        out1 = ME.cat(out1, x_list[3])
        
        out1 = self.block0(out1)
        out2 = self.relu(self.down1(self.relu(self.conv1(out1))))
        # out2 = self.dp(out2)
        
        out2 = ME.cat(out2, x_list[2])
        
        out2 = self.block1(out2)
        out3 = self.relu(self.down2(self.relu(self.conv2(out2))))
        # out3 = self.dp(out3)
        
        out3 = ME.cat(out3, x_list[1])
        
        out3 = self.block2(out3)
        out4 = self.conv3(out3)
        # out3 = self.dp(out3)
        
        out4 = ME.cat(out4, x_list[0])
        
        out4 = self.relu(self.conv_coord(out4, target))
        out4 = self.block3(out4)
        
        out5 = self.conv4(out4)
        
        return out5