from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import sys
import fastmri

from fastmri.data.subsample import MaskFunc


def image_to_kspace(tensor):
    # Apply 2D FFT
    fft_image = torch.fft.fft2(tensor)
    # Shift the zero frequency component to the center
    fft_image_shifted = torch.fft.fftshift(fft_image)
    return fft_image_shifted


def kspace_to_image(tensor):
    ifft_kspace_shifted = torch.fft.ifftshift(tensor)
    ifft_image = torch.abs(torch.fft.ifft2(ifft_kspace_shifted))
    return ifft_image


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def torch_fft_filter(image, mask):
    # Perform the FFT on the image, apply the mask, and then perform the inverse FFT
    image_fft = torch.fft.fft2(image)
    image_fft = torch.fft.fftshift(image_fft)
    filtered_fft = image_fft * mask
    filtered_img = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft)).real
    return filtered_img


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def get_mgrid_normed(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    if dim != 2:
        raise ValueError("The dim should be 2. Get {}", format(dim))
    tensors = tuple([torch.linspace(-1, 1, steps=sidelen), torch.linspace(1, -1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def get_mgrid(sidelen, dim=2):
    if dim != 2:
        raise ValueError("The dim should be 2. Get {}", format(dim))
    arrays = tuple([np.arange(sidelen), np.arange(sidelen)])
    meshgrid = np.meshgrid(*arrays, indexing='ij')
    #smeshgrid = [np.zeros((320,320)),np.zeros((320,320))]
    mgrid = np.stack(meshgrid, axis=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def positional_encoding_2d(input_tensor, num_freqs=128):
    # Extract x and y coordinates
    x = input_tensor[:, 0]
    y = input_tensor[:, 1]

    # Calculate frequencies for positional encoding
    frequencies_x = torch.pow(2.0, torch.arange(0, num_freqs).float() / 2.0)
    frequencies_y = torch.pow(2.0, (torch.arange(0, num_freqs).float() + 1.0) / 2.0)

    # Compute positional encoding for x and y coordinates
    encoding_x = torch.sin(frequencies_x * x.unsqueeze(-1))
    encoding_y = torch.cos(frequencies_y * y.unsqueeze(-1))

    # Concatenate the positional encodings along the feature dimension
    encoding = torch.cat([encoding_x, encoding_y], dim=-1)

    return encoding


def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2 ** torch.arange(min_deg, max_deg, device=x.device)
    shape = x.shape[:-1] + (-1,)
    scaled_x = (x[..., None, :] * scales[:, None]).reshape(*shape)
    # Note that we're not using safe_sin, unlike IPE.
    four_feat = torch.sin(
        torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


def normalize_minmax_perslice(data: torch.Tensor) -> torch.Tensor:
    if data.dim() == 4:
        max_values, _ = torch.max(data, dim=1, keepdim=True)
        max_values, _ = torch.max(max_values, dim=2, keepdim=True)
        max_values, _ = torch.max(max_values, dim=3, keepdim=True)
        max_values = max_values.squeeze()
        min_values, _ = torch.min(data, dim=1, keepdim=True)
        min_values, _ = torch.min(min_values, dim=2, keepdim=True)
        min_values, _ = torch.min(min_values, dim=3, keepdim=True)
        min_values = min_values.squeeze()
        return (data - max_values.view(-1, 1, 1, 1)) / (max_values - min_values).view(-1, 1, 1, 1)
    elif data.dim() == 2 or data.dim() == 3:
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    else:
        raise ValueError("The dimension is not expected, get {}", format(data.dim()))


def create_torch_convolution_kernel(mask):
    # Perform the inverse FFT to get the spatial domain kernel
    kernel = torch.fft.ifft2(torch.fft.fftshift(mask)).real
    kernel = torch.fft.ifftshift(kernel)
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel


class INRSample(NamedTuple):
    """
    A subsampled image for INR vanilla reconstruction.

    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    target: torch.Tensor
    image: torch.Tensor
    # image_fastmri: torch.Tensor
    mask: torch.Tensor
    kernel: torch.Tensor
    fname: str
    slice_num: int


class INRDataTransform:
    """
    Data Transformer for training INRVanilla models.
    """
    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        recon: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        kspace_torch = to_tensor(kspace)
        recon_torch = to_tensor(recon)
        # remove oversampled parts
        gt_normed = normalize_minmax_perslice(recon_torch)
        # apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        # we only need first and second elements, which is k-space after masking
        [masked_kspace, mask, _] = apply_mask(kspace_torch, self.mask_func, seed=seed)
        mask = mask.expand(-1, -1, 320).permute(0, 2, 1)
        # However, there are some problems in the single-coil dataset when using the image data converted from k-space.
        # Hence we use the "reconstruction_rss" in the h5 file.
        # # inverse Fourier transform to get zero filled solution
        # image_fastmri = fastmri.ifft2c(masked_kspace)
        # # absolute value
        # image_fastmri = fastmri.complex_abs(image_fastmri)
        # image_fastmri = normalize_minmax_perslice(image_fastmri)
        image = torch_fft_filter(gt_normed[None, None], mask[None]).squeeze()
        kernel = create_torch_convolution_kernel(mask[None])
        kernel = kernel[0, 0, image[None, None].shape[2] // 2, :].unsqueeze(0).unsqueeze(0).unsqueeze(2)
        image = torch.clamp(image, 0.0, 1.0)
        # image_fastmri = torch.clamp(image_fastmri, 0.0, 1.0)
        return INRSample(
            target=gt_normed,
            image=image,
            # image_fastmri=image_fastmri,
            mask=mask,
            kernel=kernel,
            fname=fname,
            slice_num=slice_num,
        )
