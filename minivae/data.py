#!/usr/bin/env python
from __future__ import annotations

import io
import json
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Protocol, Tuple, Optional

import click
import lmdb
import numpy as np
import numpy.typing as npt
import torch as th
import torchvision.datasets
import torchvision.transforms
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    # If the module is executed we need to add the parent module to the discoverable imports
    sys.path.append('.')

from minivae import common

logger = logging.getLogger(common.NAME)


CACHE_DIR = Path('~/.cache/torchvision').expanduser()
LMDB_MAP_SIZE = 1 << 40


# Disable CUDA for torch, since we only want Jax to use it
th.cuda.is_available = lambda: False


class DataConfig(Protocol):
    '''Protocol for data configuration'''
    dataset_path: Path
    shape: Tuple[int, int, int]


class DataLoaderConfig(Protocol):
    '''Protocol for data loader configuration'''
    batch_size: int
    num_workers: int


def crop_float_and_normalize(image: Image.Image,
                             size: Tuple[int, int],
                             dtype: npt.DTypeLike = np.float32,
                             ) -> np.ndarray:
    '''Center-crops, converts to float and normalizes the image'''
    # Resize
    t = torchvision.transforms.functional.resize(image, max(size))
    # Crop
    t = torchvision.transforms.functional.center_crop(t, size)
    # Add channel dimension if necessary
    x = np.asarray(t)
    if len(x.shape) == 2:
        x = x[:, :, None]
    # Normalize
    return x.astype(dtype) / 127.5 - 1.0


def transform_from_config(config: DataConfig) -> Callable[[Image.Image], np.ndarray]:
    '''Create a transform function from a configuration'''
    return partial(crop_float_and_normalize, size=config.shape[:2])


class LMDBDataset(th.utils.data.Dataset):
    '''Dataset that loads data from an LMDB database.'''

    @classmethod
    def from_config(cls,
                    config: DataConfig,
                    transform: Optional[Callable[[Image.Image], np.ndarray]] = None,
                    ) -> LMDBDataset:
        transform = transform or transform_from_config(config)
        return cls(config.dataset_path, transform)

    def __init__(self,
                 path: Path,
                 transform: Callable[[Image.Image], np.ndarray],
                 ) -> None:
        self.env = lmdb.open(str(path), readonly=True)
        self.txn = self.env.begin()
        self.keys = json.loads(self.txn.get(b'keys').decode())
        self.transform = transform

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> np.ndarray:
        key = self.keys[index]
        image_raw = self.txn.get(key.encode())
        np_file = np.load(io.BytesIO(image_raw))
        image_np = np_file[np_file.files[0]]
        if image_np.shape[-1] == 3:
            image = Image.fromarray(image_np, mode='RGB')
        elif image_np.shape[-1] == 1:
            image = Image.fromarray(image_np[:, :, 0], mode='L')
        else:
            raise ValueError(f'Unsupported number of channels: {len(image_np.shape)}')
        output = self.transform(image)
        assert len(output.shape) == 3, f'Expected 3D output, got {output.shape}'
        return output

    def get_dataloader(self,
                       batch_size: int,
                       num_workers: int,
                       ) -> th.utils.data.DataLoader:
        '''Returns a data loader for this dataset'''
        return th.utils.data.DataLoader(self,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                        drop_last=True,
                                        collate_fn=np.asarray)

    def get_dataloader_from_config(self,
                                   config: DataLoaderConfig,
                                   ) -> th.utils.data.DataLoader:
        '''Returns a data loader for this dataset'''
        return self.get_dataloader(config.batch_size, config.num_workers)


def load_dataset(name: str) -> Iterator[Dict[str, Any]]:
    name = name.strip().lower()

    def _to_uint8(x: Image.Image) -> np.ndarray:
        '''Convert an image tensor to uint8.'''
        return np.array(x).astype(np.uint8).clip(0, 255)

    images: Iterator[np.ndarray]
    if name == 'celeba':
        logger.info('Loading CelebA')
        dataset = torchvision.datasets.CelebA(
            root=str(CACHE_DIR),
            split='train',
            download=True,
            transform=_to_uint8,
        )
        logger.info('Done')
        images = (image for image, _ in dataset)
    elif name == 'mnist':
        logger.info('Loading MNIST')
        dataset = torchvision.datasets.MNIST(
            root=str(CACHE_DIR),
            train=True,
            download=True,
            transform=_to_uint8,
        )
        logger.info('Done')
        images = (image for image, _ in dataset)
    elif name == 'fashion-mnist':
        logger.info('Loading Fashion-MNIST')
        dataset = torchvision.datasets.FashionMNIST(
            root=str(CACHE_DIR),
            train=True,
            download=True,
            transform=_to_uint8,
        )
        logger.info('Done')
        images = (image for image, _ in dataset)
    elif name == 'imagenet':
        logger.info('Loading ImageNet')
        dataset = torchvision.datasets.ImageNet(
            root=str(CACHE_DIR),
            split='train',
            download=True,
            transform=_to_uint8,
        )
        logger.info('Done')
        images = (image for image, _ in dataset)
    elif name == 'dummy_rgb':
        images = (np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                  for _ in range(10))
    elif name == 'dummy_bw':
        images = (np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
                  for _ in range(10))
    else:
        raise ValueError(f'Unknown dataset: {name}')
    images = (image if len(image.shape) == 3 else image[:, :, None]
              for image in images)
    samples = (dict(image=image) for image in images)
    return samples


def store_samples(samples: Iterator[Dict[str, Any]],
                  db_path: Path,
                  ) -> None:
    '''Store samples in an LMDB database.'''
    db_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(db_path), map_size=LMDB_MAP_SIZE)
    logger.info(f'Storing samples in {db_path}')
    with env.begin(write=True) as txn:
        keys = []
        for i, sample in tqdm(enumerate(samples)):
            key = str(i)
            with io.BytesIO() as fh:
                np.savez(fh, sample['image'])
                txn.put(key.encode(), fh.getvalue())
            keys.append(key)
        txn.put('keys'.encode(), json.dumps(keys).encode())
    logger.info('Done')


def get_cli() -> click.Group:

    cli = common.get_cli_group('data')

    @cli.command('new-dataset')
    @click.option('--path', '-p', type=Path, required=True, help='Where to save the dataset')
    @click.option('--name', '-n', type=str, required=True, help='Dataset name')
    def cli_new_dataset(path: Path,
                        name: str,
                        ) -> None:
        '''Create a new dataset.'''
        samples = load_dataset(name)
        store_samples(samples, path)

    @cli.command('show-samples')
    @click.option('--path', '-p', type=Path, required=True, help='Path to the dataset')
    @click.option('--count', '-c', type=int, default=10, help='Number of samples to show')
    @click.option('--width', '-w', type=int, default=64, help='Width of the image')
    @click.option('--height', '-h', type=int, default=64, help='Height of the image')
    def cli_show_samples(path: Path,
                         count: int,
                         width: int,
                         height: int,
                         ) -> None:
        '''Show a few samples from the dataset.'''
        transform = partial(crop_float_and_normalize, size=(width, height))
        dataset = LMDBDataset(path, transform)
        indices = np.random.choice(len(dataset), count, replace=False)
        for i in indices:
            image_np = dataset[i]
            image = Image.fromarray((image_np * 127.5 + 127.5).astype(np.uint8))
            image.show()

    return cli


if __name__ == '__main__':
    get_cli()()
