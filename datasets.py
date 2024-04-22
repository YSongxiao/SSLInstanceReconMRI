import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import yaml
import json
from tqdm import tqdm


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(
    key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file: Optional; Default path config file to fetch path
            from.

    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "brain_path": "/path/to/brain",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)


class protocol_filter:
    def __init__(self, protocols: List):
        self.protocols = protocols

    def __call__(self, RawDataSample):
        if RawDataSample.metadata['acquisition'] in self.protocols:
            return True
        else:
            return False


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


def image_to_patches(image_tensor, x, y):
    if image_tensor.dim() != 3:
        raise ValueError("The shape of image_tensor should be CHW")
    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Calculate padding
    pad_h = (x - 1) // 2
    pad_w = (y - 1) // 2
    padding = (pad_w, pad_w, pad_h, pad_h)

    # Apply padding
    image_tensor_padded = torch.nn.functional.pad(image_tensor, padding, mode='reflect')

    # Use unfold to extract sliding local blocks of a x b
    patches = image_tensor_padded.unfold(2, x, 1).unfold(3, y, 1)

    # Reshape the patches to flatten each patch into a vector
    b, c, h, w, _, _ = patches.size()
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c * x * y)

    return patches


class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Optional[Callable] = None,
        p_filter: Optional[Callable] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
    ):
        """
        Args:
            root: Path to the dataset.
            paths:
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            p_filter: Optional; A callable object that takes a file's protocol
                as input and returns a boolean indicating whether the raw_sample
                should be included in the dataset.
        """

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        self.raw_samples = []
        self.protocol_filter = p_filter

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists():
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None:
            files = sorted([str(path) for path in Path(root).rglob("*.h5") if path.is_file()])
            for fname in sorted(files):
                fname = Path(fname)
                metadata, num_slices = self._retrieve_metadata(fname)

                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    if self.protocol_filter is not None:
                        if self.protocol_filter(raw_sample):
                            new_raw_samples.append(raw_sample)
                    else:
                        new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples

            dataset_cache[root] = self.raw_samples
            print(f"Saving dataset cache to {self.dataset_cache_file}.")
            with open(self.dataset_cache_file, "wb") as cache_f:
                pickle.dump(dataset_cache, cache_f)
        else:
            print(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            recon = hf["reconstruction_rss"][dataslice]

            attrs = dict(hf.attrs)
            attrs.update(metadata)

            if self.transform is None:
                raise ValueError('Transform should not be None.')
            else:
                sample = self.transform(kspace, recon, attrs, fname.name, dataslice)
        return sample


class ImageFitting(torch.utils.data.Dataset):
    def __init__(self, sidelength, sample, pos_dim=128):
        super().__init__()
        self.d_pixels = sample.image[None].permute(1, 2, 0).view(-1, 1)
        self.pixels = sample.target[None].permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid_normed(sidelength, 2)
        self.d_image = sample.image[None]
        self.gt_image = sample.target[None]
        self.mask = sample.mask[None]
        self.pos_encoding = pos_enc(self.coords, min_deg=0, max_deg=pos_dim)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.pos_encoding, self.d_pixels, self.pixels, self.d_image, self.gt_image, self.mask


def create_data_loader(dataset, batch_size, num_workers, sampler=None, is_train=True):
    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )
    return dataloader
