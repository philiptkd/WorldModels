import numpy as np
import torch
import torch.nn as nn

def conv(x, kernel_sizes=[4, 4, 4, 4], strides=[2, 2, 2, 2],
         out_channels=[32, 64, 128, 256]):
    _, in_channels, in_height, in_width = x.shape
    assert len(kernel_sizes) == len(strides) == len(out_channels)

    output = x
    for i in range(len(strides)):
        net = nn.Conv2d(in_channels, out_channels[i], kernel_sizes[i], strides[i])
        output = net(output)
        in_channels = out_channels[i]

        print(output.shape)

x = torch.ones((1, 3, 64, 64))
conv(x)

def deconv(z, kernel_sizes=[5, 5, 6, 6], strides = [2, 2, 2, 2],
           out_channels=[128, 64, 32, 3]):
    _, in_channels, in_height, in_width = z.shape
    assert len(kernel_sizes) == len(strides) == len(out_channels)

    output = z
    for i in range(len(strides)):
        net = nn.ConvTranspose2d(in_channels, out_channels[i], kernel_sizes[i], strides[i])
        output = net(output)
        in_channels = out_channels[i]

        print(output.shape)

z = torch.ones((1, 1024, 1, 1))
deconv(z)
