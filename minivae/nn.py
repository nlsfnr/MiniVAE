'''Implementation of the model and its components.'''
from __future__ import annotations

import logging
from functools import partial
from typing import List, NamedTuple, Protocol, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, PRNGKey

from minivae import common

logger = logging.getLogger(common.NAME)


class ConvBlock(hk.Module):

    def __init__(self,
                 size: int,
                 kernel_shape: int,
                 stride: int,
                 dropout: float,
                 name=None,
                 ) -> None:
        super().__init__(name=name)
        self.size = size
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.dropout = dropout

    @partial(common.consistent_axes, inherit='B')
    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> Array:
        y = hk.LayerNorm(-1, True, True, name='ln1')(x)
        y = jax.nn.gelu(y)
        y = hk.Conv2D(self.size,
                      kernel_shape=self.kernel_shape,
                      stride=self.stride,
                      with_bias=False,
                      name='conv_1',
                      )(y)
        common.assert_shape(y, 'B H W C')
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        y = hk.LayerNorm(-1, True, True, name='ln2')(y)
        y = jax.nn.gelu(y)
        y = hk.Conv2D(self.size,
                      kernel_shape=self.kernel_shape,
                      stride=1,
                      with_bias=False,
                      name='conv_2',
                      )(y)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        if x.shape != y.shape:
            x = hk.Conv2D(self.size,
                          kernel_shape=self.kernel_shape,
                          stride=self.stride,
                          with_bias=False,
                          name='skip_conv',
                          )(x)
        common.assert_shape(x, 'B H W C')
        return x + y


class ConvTransposeBlock(hk.Module):

    def __init__(self,
                 size: int,
                 kernel_shape: int,
                 stride: int,
                 dropout: float,
                 name=None,
                 ) -> None:
        super().__init__(name=name)
        self.size = size
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.dropout = dropout

    @partial(common.consistent_axes, inherit='B')
    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> Array:
        y = hk.LayerNorm(-1, True, True)(x)
        y = jax.nn.gelu(y)
        y = hk.Conv2DTranspose(self.size,
                               kernel_shape=self.kernel_shape,
                               stride=self.stride,
                               with_bias=False,
                               name='conv_1',
                               )(y)
        common.assert_shape(y, 'B H W C')
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        y = hk.LayerNorm(-1, True, True)(y)
        y = jax.nn.gelu(y)
        y = hk.Conv2DTranspose(self.size,
                               kernel_shape=self.kernel_shape,
                               stride=1,
                               with_bias=False,
                               name='conv_2',
                               )(y)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        if x.shape != y.shape:
            x = hk.Conv2DTranspose(self.size,
                                   kernel_shape=self.kernel_shape,
                                   stride=self.stride,
                                   with_bias=False,
                                   name='skip_conv',
                                   )(x)
        common.assert_shape(x, 'B H W C')
        return x + y


class VAEEncoderOutput(NamedTuple):
    mean: Array
    stddev: Array


class VAEEncoder(hk.Module):
    '''VAE encoder module.'''

    def __init__(self,
                 sizes: Sequence[int],
                 strides: Sequence[int],
                 out_size: int,
                 dropout: float,
                 name=None,
                 ) -> None:
        super().__init__(name=name)
        self.sizes = sizes
        self.strides = strides
        self.out_size = out_size
        self.dropout = dropout

    @partial(common.consistent_axes, inherit='B H W LH LW LC')
    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> VAEEncoderOutput:
        '''Forward pass.'''
        x = hk.Conv2D(self.sizes[0],
                      kernel_shape=1,
                      stride=1,
                      name='conv_in',
                      )(x)
        common.assert_shape(x, 'B H W C')
        x = hk.get_parameter('pos_embedding',
                             x.shape[1:],
                             x.dtype,
                             init=hk.initializers.RandomNormal(0.01))[None, :, :, :] + x
        for i, (size, stride) in enumerate(zip(self.sizes, self.strides)):
            x = ConvBlock(size,
                          kernel_shape=3,
                          stride=stride,
                          dropout=self.dropout,
                          name=f'block_{i}',
                          )(x, is_training)
        x = hk.LayerNorm(-1, True, True)(x)
        x = jax.nn.gelu(x)
        x = hk.Conv2D(self.out_size * 2,
                      kernel_shape=1,
                      stride=1,
                      name='conv_out',
                      w_init=hk.initializers.VarianceScaling(0.01),
                      )(x)
        mean, log_stddev = jnp.split(x, 2, axis=-1)
        stddev = jnp.exp(log_stddev)
        common.assert_shape(mean, 'B LH LW LC')
        common.assert_shape(stddev, 'B LH LW LC')
        return VAEEncoderOutput(mean, stddev)


class VAEDecoder(hk.Module):
    '''VAE decoder module.'''

    def __init__(self,
                 sizes: Sequence[int],
                 strides: Sequence[int],
                 out_size: int,
                 dropout: float,
                 name=None,
                 ) -> None:
        super().__init__(name=name)
        self.sizes = sizes
        self.strides = strides
        self.out_size = out_size
        self.dropout = dropout

    @partial(common.consistent_axes, inherit='B H W LH LW LC')
    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> Array:
        '''Forward pass.'''
        common.assert_shape(x, 'B LH LW LC')
        x = hk.Conv2D(self.sizes[0],
                      kernel_shape=1,
                      stride=1,
                      name='conv_in',
                      )(x)
        x = hk.get_parameter('pos_embedding',
                             x.shape[1:],
                             x.dtype,
                             init=hk.initializers.RandomNormal(0.01))[None] + x
        for i, (size, stride) in enumerate(zip(self.sizes, self.strides)):
            x = ConvTransposeBlock(size,
                                   kernel_shape=3,
                                   stride=stride,
                                   dropout=self.dropout,
                                   name=f'block_{i}',
                                   )(x, is_training)
        common.assert_shape(x, 'B H W C')
        x = hk.LayerNorm(-1, True, True)(x)
        x = jax.nn.gelu(x)
        x = hk.Conv2D(self.out_size,
                      kernel_shape=1,
                      stride=1,
                      name='conv_out',
                      )(x)
        x = jax.nn.tanh(x)
        return x


class ModelConfig(Protocol):
    shape: Tuple[int, int, int]
    encoder_sizes: List[int]
    encoder_strides: List[int]
    decoder_sizes: List[int]
    decoder_strides: List[int]
    latent_size: int
    dropout: float


class VAEOutput(NamedTuple):
    x_hat: Array
    reconstruction_loss: Array
    kl_loss: Array
    mean: Array
    stddev: Array


class VAE(hk.Module):

    def __init__(self,
                 encoder_sizes: Sequence[int],
                 encoder_strides: Sequence[int],
                 decoder_sizes: Sequence[int],
                 decoder_strides: Sequence[int],
                 latent_size: int,
                 image_channels: int,
                 dropout: float,
                 name=None,
                 ) -> None:
        super().__init__(name=name)
        self.encoder_sizes = encoder_sizes
        self.encoder_strides = encoder_strides
        self.decoder_sizes = decoder_sizes
        self.decoder_strides = decoder_strides
        self.latent_size = latent_size
        self.image_channels = image_channels
        self.dropout = dropout

    @classmethod
    def from_config(cls,
                    config: ModelConfig,
                    ) -> VAE:
        return cls(encoder_sizes=config.encoder_sizes,
                   encoder_strides=config.encoder_strides,
                   decoder_sizes=config.decoder_sizes,
                   decoder_strides=config.decoder_strides,
                   latent_size=config.latent_size,
                   image_channels=config.shape[-1],
                   dropout=config.dropout)

    @classmethod
    def get_params(cls,
                   config: ModelConfig,
                   rng: PRNGKey,
                   ) -> ArrayTree:
        inputs = jnp.zeros((1, *config.shape), dtype=jnp.float32)
        fn = lambda: cls.from_config(config)(inputs, True)
        params = hk.transform(fn).init(rng)
        params_n = hk.data_structures.tree_size(params)
        params_mb = round(hk.data_structures.tree_bytes(params) / 1e6, 2)
        logger.info(f'Parameters: {params_n:,} ({params_mb:.2f} MB)')
        return params

    @partial(common.consistent_axes, inherit='B H W C')
    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> VAEOutput:
        # Mean and stddev of latent space.
        mean, stddev = self.encode(x, is_training)
        common.assert_shape(mean, 'B LH LW LC')
        common.assert_shape(stddev, 'B LH LW LC')
        # Sample from latent space.
        z = mean + stddev * jax.random.normal(hk.next_rng_key(), stddev.shape)
        # Decode latent space.
        x_hat = self.decode(z, is_training)
        common.assert_shape(stddev, 'B H W C')
        # Reconstruction loss. We use MSE here.
        reconstruction_loss = jnp.mean((x - x_hat) ** 2)
        # KL divergence. Appendix B from Kingma & Welling (2013).
        var = stddev ** 2
        kl_loss = -0.5 * jnp.mean(1 + jnp.log(var) - mean ** 2 - var)
        return VAEOutput(x_hat=x_hat,
                         reconstruction_loss=reconstruction_loss,
                         kl_loss=kl_loss,
                         mean=mean,
                         stddev=stddev)

    def encode(self,
               x: Array,
               is_training: bool,
               ) -> VAEEncoderOutput:
        '''Encode input.'''
        mean, stddev = VAEEncoder(sizes=self.encoder_sizes,
                                  strides=self.encoder_strides,
                                  out_size=self.latent_size,
                                  dropout=self.dropout,
                                  name='encoder')(x, is_training)
        assert mean.dtype == jnp.float32
        assert stddev.dtype == jnp.float32
        return VAEEncoderOutput(mean, stddev)

    def decode(self,
               z: Array,
               is_training: bool,
               ) -> Array:
        '''Decode latent space.'''
        x_hat = VAEDecoder(sizes=self.decoder_sizes,
                           strides=self.decoder_strides,
                           out_size=self.image_channels,
                           dropout=self.dropout,
                           name='decoder')(z, is_training)
        assert x_hat.dtype == jnp.float32
        return x_hat
