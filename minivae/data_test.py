from functools import partial
from pathlib import Path

import numpy as np
import pytest

from minivae import data


@pytest.mark.parametrize('dataset_name', ['dummy_rgb', 'dummy_bw'])
def test_load_dataset(dataset_name: str) -> None:
    samples = data.load_dataset(dataset_name)
    sample = next(samples)
    assert sample['image'].shape[-1] == 3 if dataset_name == 'dummy_rgb' else 1
    assert len(sample['image'].shape) == 3
    assert sample['image'].dtype == 'uint8'


@pytest.mark.parametrize('dataset_name', ['dummy_rgb', 'dummy_bw'])
def test_lmdbdataset(tmpdir: Path, dataset_name: str) -> None:
    tmpdir = Path(tmpdir)
    samples = data.load_dataset(dataset_name)
    data.store_samples(samples, tmpdir)
    transform = partial(data.crop_float_and_normalize, size=(32, 32))
    dataset = data.LMDBDataset(tmpdir, transform)
    sample = dataset[0]
    assert sample.shape == (32, 32, 3 if dataset_name == 'dummy_rgb' else 1)
    assert sample.dtype == np.float32
    assert (-1.0 <= sample).all()
    assert (sample <= 1.0).all()
    assert isinstance(sample, np.ndarray)


@pytest.mark.parametrize('dataset_name', ['dummy_rgb', 'dummy_bw'])
def test_lmdbdataset_get_dataloader(tmpdir: Path, dataset_name: str) -> None:
    tmpdir = Path(tmpdir)
    samples = data.load_dataset(dataset_name)
    data.store_samples(samples, tmpdir)
    transform = partial(data.crop_float_and_normalize, size=(32, 32))
    dataset = data.LMDBDataset(tmpdir, transform)
    dataloader = dataset.get_dataloader(batch_size=4, num_workers=4)
    sample = next(iter(dataloader))
    assert sample.shape == (4, 32, 32, 3 if dataset_name == 'dummy_rgb' else 1)
    assert sample.dtype == np.float32
    assert (-1.0 <= sample).all()
    assert (sample <= 1.0).all()
    assert isinstance(sample, np.ndarray)
