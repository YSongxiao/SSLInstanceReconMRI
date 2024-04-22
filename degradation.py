import torch
import torch.fft as fft
from skimage.registration import phase_cross_correlation
import numpy as np


def align_images(image1, image2):
    # Compute the shift between the two images
    shift, error, diffphase = phase_cross_correlation(image1, image2)
    # Shift the second image to align with the first image
    shifted_image2 = np.roll(image2, shift.astype(int), axis=(0, 1))
    return shifted_image2


def get_shift(image1, image2):
    # Compute the shift between the two images
    shift, error, diffphase = phase_cross_correlation(image1, image2)
    # Shift the second image to align with the first image
    # shifted_image2 = np.roll(image2, shift.astype(int), axis=(0, 1))
    # print(shift)
    return tuple(shift.astype(int))


def get_shift_multi_channel(image1, image2):
    # 确保两个图像具有相同数量的通道
    assert image1.shape[0] == image2.shape[0], "Images must have the same number of channels"

    # 初始化存储每个通道位移的列表
    shifts = []

    # 对每个通道计算位移
    for channel in range(image1.shape[0]):
        # 选取当前通道
        channel_image1 = image1[channel, ...]
        channel_image2 = image2[channel, ...]

        # 计算位移
        shift, error, diffphase = phase_cross_correlation(channel_image1, channel_image2)

        # 添加到列表
        shifts.append(shift.astype(int))

    return shifts


def roll_channels(image, shifts):
    """
    Roll each channel of a multi-channel image with corresponding shifts.

    :param image: A multi-channel image tensor of shape (C, H, W),
                  where C is the number of channels.
    :param shifts: A list of tuples, each containing two integers,
                   representing the vertical and horizontal shifts
                   for each channel.
    :return: The rolled image tensor.
    """
    # 确保图像和位移列表具有相同数量的通道
    assert image.shape[0] == len(shifts), "The number of shifts must match the number of channels in the image"

    # 创建一个空的列表来存储每个通道的结果
    rolled_channels = []

    # 对每个通道应用滚动变换
    for channel, shift in zip(image, shifts):
        # 应用滚动
        rolled_channel = torch.roll(channel, shifts=tuple(shift), dims=(0, 1))

        # 添加到结果列表
        rolled_channels.append(rolled_channel)

    # 将结果堆叠回一个多通道图像
    rolled_image = torch.stack(rolled_channels, dim=0)

    return rolled_image
