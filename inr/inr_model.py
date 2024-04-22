import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

from degradation import align_images
from transforms import image_to_kspace, kspace_to_image
import fastmri


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, use_group_norm=False):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.norm1 = nn.GroupNorm(32, out_channels) if use_group_norm else nn.Identity()
        # self.activation = SwishActivation()
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.norm2 = nn.GroupNorm(32, out_channels) if use_group_norm else nn.Identity()

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.norm1(out)
        # out = self.activation(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += shortcut
        out = self.relu(out)
        # TODO： activation
        return out


class ResidualEncoder(nn.Module):
    '''
    Contains x scale-embedded residual blocks.
    All convolutional layers have 5x5-sized kernels for a large receptive field and 64 feature channels.
    '''
    def __init__(self, num_resblocks, in_channels=1, channels=64, final_dim=128):
        super().__init__()
        self.in_block = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=5, stride=1, padding=2)
        self.blocks = nn.Sequential(*[ResidualBlock(in_channels=channels, out_channels=channels) for i in range(num_resblocks)])
        self.out_block = nn.Conv2d(in_channels=channels, out_channels=final_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.in_block(x)
        for block in self.blocks:
            x = block(x)
        out = self.out_block(x)
        return out


class SelfSiren(nn.Module):
    def __init__(self, in_features, out_features, kernel, hidden_features=256, hidden_layers=3, outermost_linear=True):
        super().__init__()
        self.img_siren = Siren(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                          hidden_layers=hidden_layers, outermost_linear=outermost_linear)
        self.degrad_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 320), padding='same',
                                     padding_mode='circular')
        self.degrad_conv.weight = nn.Parameter(kernel)
        self.degrad_conv.bias = nn.Parameter(torch.zeros(1))  # Setting the bias to zero
        for param in self.degrad_conv.parameters():
            param.requires_grad = False

    def forward(self, coords):
        output, _ = self.img_siren(coords)
        conv_img = self.degrad_conv(output.view(1, 1, 320, 320))
        return output, conv_img


class SelfResidualSiren(nn.Module):
    def __init__(self, in_features, out_features, kernel, hidden_features=256, hidden_layers=3, outermost_linear=True):
        super().__init__()
        self.img_siren = Siren(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                          hidden_layers=hidden_layers, outermost_linear=outermost_linear)
        self.degrad_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 320), padding='same',
                                     padding_mode='circular')
        self.degrad_conv.weight = nn.Parameter(kernel)
        self.degrad_conv.bias = nn.Parameter(torch.zeros(1))  # Setting the bias to zero
        for param in self.degrad_conv.parameters():
            param.requires_grad = False

    def forward(self, coords, d_pixels):
        model_input = torch.cat((coords, d_pixels), dim=-1)
        residual, _ = self.img_siren(model_input)
        output = residual + d_pixels
        conv_img = self.degrad_conv(output.view(1, 1, 320, 320))
        return output, conv_img, residual


class SelfSirenWithFeature(nn.Module):
    def __init__(self, num_resblocks, encoder_in_features, encoder_out_features, mlp_in_features, out_features, kernel, hidden_features=256, hidden_layers=3, outermost_linear=True):
        super().__init__()
        self.feature_dim = encoder_out_features
        self.encoder = ResidualEncoder(num_resblocks, encoder_in_features, encoder_out_features)
        self.img_siren = Siren(in_features=mlp_in_features+encoder_out_features, out_features=out_features,
                               hidden_features=hidden_features, hidden_layers=hidden_layers,
                               outermost_linear=outermost_linear)
        self.degrad_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 320), padding='same',
                                     padding_mode='circular')
        self.degrad_conv.requires_grad_(False)
        self.degrad_conv.weight = nn.Parameter(kernel)
        self.degrad_conv.bias = nn.Parameter(torch.zeros(1))  # Setting the bias to zero

    def forward(self, coords, image):
        features = self.encoder(image).view(320, 320, self.feature_dim).view(-1, self.feature_dim)[None]
        output, _ = self.img_siren(torch.cat((coords, features), dim=-1))
        conv_img = self.degrad_conv(output.view(1, 1, 320, 320))
        return output, conv_img


class MulticoilSelfSiren(nn.Module):
    def __init__(self, in_features, out_features, kernel, num_coils, hidden_features=256, hidden_layers=3, outermost_linear=True):
        super().__init__()
        self.num_coils = num_coils
        self.img_siren = Siren(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                          hidden_layers=hidden_layers, outermost_linear=outermost_linear)
        self.degrad_conv = nn.Conv2d(in_channels=num_coils, out_channels=num_coils, kernel_size=(1, 320), padding='same', groups=15,
                             padding_mode='circular')
        self.degrad_conv.weight = nn.Parameter(kernel.expand(15, -1, -1, -1))
        self.degrad_conv.bias = nn.Parameter(torch.zeros(15))  # Setting the bias to zero
        self.degrad_conv_relaxed = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 320), padding='same',
                             padding_mode='circular')
        self.degrad_conv_relaxed.weight = nn.Parameter(kernel)
        self.degrad_conv_relaxed.bias = nn.Parameter(torch.zeros(1))  # Setting the bias to zero
        for param in self.degrad_conv.parameters():
            param.requires_grad = False
        for param in self.degrad_conv_relaxed.parameters():
            param.requires_grad = False

    def forward(self, coords):
        output, _ = self.img_siren(coords)
        conv_img = self.degrad_conv((output.view(320, 320, self.num_coils).permute(2, 0, 1))[None])
        conv_img_relaxed = self.degrad_conv_relaxed(fastmri.rss(output.view(320, 320, self.num_coils).permute(2, 0, 1))[None])
        return output, conv_img, conv_img_relaxed


class MulticoilResidualSelfSiren(nn.Module):
    def __init__(self, in_features, out_features, kernel, num_coils, hidden_features=256, hidden_layers=3, outermost_linear=True):
        super().__init__()
        self.num_coils = num_coils
        self.img_siren = Siren(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                          hidden_layers=hidden_layers, outermost_linear=outermost_linear)
        self.degrad_conv = nn.Conv2d(in_channels=num_coils, out_channels=num_coils, kernel_size=(1, 320), padding='same', groups=15,
                             padding_mode='circular')
        self.degrad_conv.weight = nn.Parameter(kernel.expand(15, -1, -1, -1))
        self.degrad_conv.bias = nn.Parameter(torch.zeros(15))  # Setting the bias to zero
        self.degrad_conv_relaxed = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 320), padding='same',
                             padding_mode='circular')
        self.degrad_conv_relaxed.weight = nn.Parameter(kernel)
        self.degrad_conv_relaxed.bias = nn.Parameter(torch.zeros(1))  # Setting the bias to zero
        for param in self.degrad_conv.parameters():
            param.requires_grad = False
        for param in self.degrad_conv_relaxed.parameters():
            param.requires_grad = False

    def forward(self, coords, pixels):
        model_input = torch.cat((coords, pixels), dim=-1)
        output, _ = self.img_siren(model_input)
        pred = pixels + output
        conv_img = self.degrad_conv(pred.view(320, 320, self.num_coils).permute(2, 0, 1)[None])
        conv_img_relaxed = self.degrad_conv_relaxed(fastmri.rss(pred.view(320, 320, self.num_coils).permute(2, 0, 1))[None])
        return pred, conv_img, conv_img_relaxed


class SelfSirenKspaceMasking(nn.Module):
    def __init__(self, in_features, out_features, mask, hidden_features=256, hidden_layers=3, outermost_linear=True):
        super().__init__()
        self.img_siren = Siren(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                          hidden_layers=hidden_layers, outermost_linear=outermost_linear)
        self.mask = mask

    def forward(self, coords):
        output, _ = self.img_siren(coords)
        output_img = output.view(320, 320)
        output_kspace = image_to_kspace(output_img)
        if output_kspace.device != self.mask.device:
            self.mask = self.mask.to(output_kspace.device)
        masked_output_kspace = output_kspace * self.mask
        masked_output_img = kspace_to_image(masked_output_kspace)
        return output, masked_output_img


class SelfUnrollSiren(nn.Module):
    def __init__(self, iter_num, in_features, out_features, mask, kernel, hidden_features=256, hidden_layers=3, outermost_linear=True):
        super().__init__()
        self.sirens = nn.ModuleList(Siren(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                          hidden_layers=hidden_layers, outermost_linear=outermost_linear) for i in range(iter_num))
        self.mask = mask.cuda()
        self.degrad_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 320), padding='same',
                                     padding_mode='circular')
        self.degrad_conv.weight = nn.Parameter(kernel)
        self.degrad_conv.bias = nn.Parameter(torch.zeros(1))  # Setting the bias to zero
        for param in self.degrad_conv.parameters():
            param.requires_grad = False

    def forward(self, pixels, coords, d_image):
        ori_kspace = image_to_kspace(d_image) * self.mask
        output = []
        conv_imgs = []
        for siren in self.sirens:
            pixels, _ = siren(torch.cat((pixels, coords), dim=-1))
            pred_img = pixels.view(1, 320, 320)
            pred_kspace = image_to_kspace(pred_img) * (1 - self.mask)
            pred_kspace = pred_kspace + ori_kspace
            pred_img = (kspace_to_image(pred_kspace)).view(1, 320, 320)
            pixels = pred_img.permute(1, 2, 0).view(1, -1, 1)
            output.append(pixels)
            conv_imgs.append(self.degrad_conv(pixels.view(1, 1, 320, 320)))
        return output, conv_imgs
