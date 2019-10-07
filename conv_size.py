# used to test different ConvVAE parameters for a smaller image size (10x10)

import numpy as np
import torch
import torch.nn as nn

# original
#imsize = 48
#kernel_sizes = [5, 3, 3]
#strides = [3, 2, 2]

#deconv_insize = 3
#deconv_kernel_sizes = [3, 3, 6]
#deconv_strides = [2, 2, 3]

# smaller
imsize = 10
kernel_sizes = [2,2]
strides = [2,2]

deconv_insize = 2
deconv_kernel_sizes = [2, 2]
deconv_strides = [3, 2]

def get_conv_out_size(imsize, kernel_sizes, strides):
    assert len(kernel_sizes) == len(strides)
    n_layers = len(strides)

    size = imsize
    for i in range(n_layers):
        size = np.floor((size - kernel_sizes[i])/strides[i] + 1)
        print(size)


def get_conv_out_size2(imsize, kernel_sizes, strides):
    dummy_input = torch.ones((1, 3, imsize, imsize))
    in_channels = out_channels = 3 # simplicity

    output = dummy_input
    print(output.shape)

    for i in range(len(strides)):
        net = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i], stride=strides[i])
        output = net(output)
        print(output.shape)

def get_deconv_out_size(insize, kernel_sizes, strides):
    assert len(kernel_sizes) == len(strides)
    n_layers = len(strides)

    size = insize
    for i in range(n_layers):
        size = (size - 1)*strides[i] + kernel_sizes[i]
        print(size)

def get_deconv_out_size2(insize, kernel_sizes, strides):
    dummy_input = torch.ones((1, 3, insize, insize))
    in_channels = out_channels = 3 # simplicity

    output = dummy_input
    print(output.shape)

    for i in range(len(strides)):
        net = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_sizes[i], stride=strides[i])
        output = net(output)
        print(output.shape)

get_deconv_out_size2(deconv_insize, deconv_kernel_sizes, deconv_strides)