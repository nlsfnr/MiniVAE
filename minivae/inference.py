#!/usr/bin/env python3
'''Inference functions for the model.'''
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
import random
import math
from typing import Optional, Protocol, List, Tuple

import chex
import click
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from PIL import Image

if __name__ == '__main__':
    # If the module is executed we need to add the parent module to the discoverable imports
    import sys
    sys.path.append('.')

from minivae import common, data, nn

logger = logging.getLogger(common.NAME)


class InferenceConfig(nn.ModelConfig, Protocol):
    tokenizer_path: Path


def generate(config: InferenceConfig,
             params: chex.ArrayTree,
             z: Array,
             use_jit: bool = False,
             ) -> Array:
    '''Sample from the model.'''

    def model_fn() -> Array:
        model = nn.VAE.from_config(config)
        return model.decode(z, is_training=False)

    # Execution
    with jax.disable_jit(not use_jit):
        forward = partial(hk.without_apply_rng(hk.transform(model_fn)).apply, params)
        outputs = forward()
        return outputs


def recreate(config: InferenceConfig,
             params: chex.ArrayTree,
             x: Array,
             rng: chex.PRNGKey,
             use_jit: bool = False,
             ) -> nn.VAEOutput:
    '''Reconstruct an image.'''

    def model_fn() -> nn.VAEOutput:
        model = nn.VAE.from_config(config)
        return model(x, is_training=False)

    # Execution
    with jax.disable_jit(not use_jit):
        forward = partial(hk.transform(model_fn).apply, params)
        outputs = forward(rng)
        return outputs



def get_cli() -> click.Group:
    '''Get the command line interface for this module.'''

    class Config(common.YamlConfig):

        # Model config
        encoder_sizes: List[int]
        decoder_sizes: List[int]
        latent_size: int
        shape: Tuple[int, int, int]
        dropout: float = 0.1

    cli = common.get_cli_group('inference')

    @cli.command('generate')
    @click.option('--load-from', '-l', type=Path,
                  help='Path to the checkpoint to use for generation')
    @click.option('--out', '-o', type=Path, default=None,
                  help='Path to the output image')
    @click.option('--seed', '-s', type=int, default=None, help='Random seed')
    def cli_generate(load_from: Path,
                     out: Optional[Path],
                     seed: Optional[int],
                     ) -> None:
        '''Generate a new image from a model.'''
        rngs = common.get_rngs(seed)
        checkpoint = common.load_checkpoint(load_from, config_class=Config)
        config = checkpoint['config']
        params = checkpoint['params']
        height, width, _ = config.shape
        fraction = 0.5 ** len(config.encoder_sizes)
        h, w, c = int(height * fraction), int(width * fraction), config.latent_size
        z = jax.random.normal(next(rngs), (1, h, w, c))
        x = generate(config, params, z, use_jit=False)[0]
        if x.shape[-1] == 1:
            x = x[..., 0]
        image = Image.fromarray((np.asarray(x) * 127.5 + 127.5).astype(np.uint8), 'L')
        image.show()
        if out is None:
            image.show()
        else:
            image.save(out)


    @cli.command('recreate')
    @click.option('--load-from', '-l', type=Path,
                  help='Path to the checkpoint to use for generation')
    @click.option('--image-path', '-i', type=Path, default=None,
                  help='Path to the image to recreate. If not provided, a random image will be '
                  'used.')
    @click.option('--dataset-path', '-d', type=Path, default=None,
                  help='Path to the dataset to use for random image selection.')
    @click.option('--out', '-o', type=Path, default=None,
                  help='Path to the output image')
    @click.option('--seed', '-s', type=int, default=None, help='Random seed')
    def cli_recreate(load_from: Path,
                     image_path: Optional[Path],
                     dataset_path: Optional[Path],
                     out: Optional[Path],
                     seed: Optional[int],
                     ) -> None:
        '''Recreate an image from a model.'''
        rngs = common.get_rngs(seed)
        checkpoint = common.load_checkpoint(load_from, config_class=Config)
        config = checkpoint['config']
        params = checkpoint['params']
        transform = data.transform_from_config(config)
        if dataset_path is not None:
            dataset = data.LMDBDataset(dataset_path, transform=transform)
            idx = random.randrange(0, len(dataset))
            inputs = jnp.asarray(dataset[idx])
        elif image_path is not None:
            image = Image.open(image_path)
            inputs = jnp.asarray(transform(image))
        else:
            raise ValueError('Either image-path or dataset-path must be provided.')
        output = recreate(config, params, inputs, next(rngs), use_jit=False)
        both = jnp.concatenate([inputs, output.x_hat], axis=1)
        if both.shape[-1] == 1:
            both = both[..., 0]
        image = Image.fromarray((np.asarray(both) * 127.5 + 127.5).astype(np.uint8), 'L')
        image.show()
        if out is None:
            image.show()
        else:
            image.save(out)


    return cli


if __name__ == '__main__':
    get_cli()()
