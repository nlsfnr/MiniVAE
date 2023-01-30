from __future__ import annotations

from typing import List, NamedTuple, Protocol, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, PRNGKey


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

    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> Array:
        y = hk.LayerNorm(-1, True, True, name='ln1')(x)
        y = jax.nn.gelu(y)
        y = hk.Conv2D(self.size,
                      kernel_shape=self.kernel_shape,
                      stride=self.stride,
                      name='conv_1',
                      )(y)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        y = hk.LayerNorm(-1, True, True, name='ln2')(y)
        y = jax.nn.gelu(y)
        y = hk.Conv2D(self.size,
                      kernel_shape=self.kernel_shape,
                      stride=1,
                      name='conv_2',
                      )(y)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        if self.stride != 1:
            x = hk.Conv2D(self.size,
                          kernel_shape=self.kernel_shape,
                          stride=self.stride,
                          name='skip_conv',
                          )(x)
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

    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> Array:
        y = hk.LayerNorm(-1, True, True)(x)
        y = jax.nn.gelu(y)
        y = hk.Conv2DTranspose(self.size,
                               kernel_shape=self.kernel_shape,
                               stride=1,
                               )(y)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        y = hk.LayerNorm(-1, True, True)(y)
        y = jax.nn.gelu(y)
        y = hk.Conv2DTranspose(self.size,
                               kernel_shape=self.kernel_shape,
                               stride=self.stride
                               )(y)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        if self.stride != 1:
            x = hk.Conv2DTranspose(self.size,
                                   kernel_shape=self.kernel_shape,
                                   stride=self.stride)(x)
        return x + y


class VAEEncoderOutput(NamedTuple):
    mean: Array
    stddev: Array


class VAEEncoder(hk.Module):
    '''VAE encoder module.'''

    def __init__(self,
                 sizes: Sequence[int],
                 out_size: int,
                 dropout: float,
                 name=None,
                 ) -> None:
        super().__init__(name=name)
        self.sizes = sizes
        self.out_size = out_size
        self.dropout = dropout

    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> VAEEncoderOutput:
        '''Forward pass.'''
        x = hk.Conv2D(self.sizes[0],
                      kernel_shape=1,
                      stride=1)(x)
        for i, size in enumerate(self.sizes):
            x = ConvBlock(size,
                          kernel_shape=3,
                          stride=2,
                          dropout=self.dropout,
                          name=f'block_{i}',
                          )(x, is_training)
        x = hk.LayerNorm(-1, True, True)(x)
        x = jax.nn.gelu(x)
        x = hk.Conv2D(self.out_size * 2,
                      kernel_shape=1,
                      stride=1)(x)
        mean, stddev = jnp.split(x, 2, axis=-1)
        return VAEEncoderOutput(mean, stddev)


class VAEDecoder(hk.Module):
    '''VAE decoder module.'''

    def __init__(self,
                 sizes: Sequence[int],
                 out_size: int,
                 dropout: float,
                 name=None,
                 ) -> None:
        super().__init__(name=name)
        self.sizes = sizes
        self.out_size = out_size
        self.dropout = dropout

    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> Array:
        '''Forward pass.'''
        x = hk.Conv2D(self.sizes[0],
                      kernel_shape=1,
                      stride=1)(x)
        for i, size in enumerate(self.sizes):
            x = ConvTransposeBlock(size,
                                   kernel_shape=3,
                                   stride=2,
                                   dropout=self.dropout,
                                   name=f'block_{i}',
                                   )(x, is_training)
        x = hk.LayerNorm(-1, True, True)(x)
        x = jax.nn.gelu(x)
        x = hk.Conv2D(self.out_size,
                      kernel_shape=1,
                      stride=1)(x)
        x = jax.nn.tanh(x)
        return x


class ModelConfig(Protocol):
    shape: Tuple[int, int, int]
    encoder_sizes: List[int]
    decoder_sizes: List[int]
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
                 decoder_sizes: Sequence[int],
                 latent_size: int,
                 image_channels: int,
                 dropout: float,
                 name=None,
                 ) -> None:
        super().__init__(name=name)
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.latent_size = latent_size
        self.image_channels = image_channels
        self.dropout = dropout

    @classmethod
    def from_config(cls,
                    config: ModelConfig,
                    ) -> VAE:
        return cls(encoder_sizes=config.encoder_sizes,
                   decoder_sizes=config.decoder_sizes,
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
        return hk.transform(fn).init(rng)

    def __call__(self,
                 x: Array,
                 is_training: bool,
                 ) -> VAEOutput:
        # Mean and stddev of latent space.
        mean, stddev = self.encode(x, is_training)
        # Sample from latent space.
        z = mean + stddev * jax.random.normal(hk.next_rng_key(), stddev.shape)
        # Decode latent space.
        x_hat = self.decode(z, is_training)
        # Reconstruction loss.
        reconstruction_loss = jnp.mean((x - x_hat) ** 2)
        # KL divergence.
        # TODO: KL Div is not correct.
        # kl_loss = -0.5 * jnp.mean(1. + jnp.log(var + 1e-6) - var + jnp.square(mean))
        kl_loss = jnp.mean((jnp.abs(stddev) - 1.) ** 2) + jnp.mean(mean ** 2)
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
        x_hat = VAEDecoder(self.decoder_sizes,
                           self.image_channels,
                           dropout=self.dropout,
                           name='decoder')(z, is_training)
        assert x_hat.dtype == jnp.float32
        return x_hat
