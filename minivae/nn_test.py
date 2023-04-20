import haiku as hk
import jax.numpy as jnp

from minivae import nn


def test_conv_block_call() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        block = nn.ConvBlock(size=16, kernel_shape=3, stride=2, dropout=0.1)
        x = jnp.ones((1, 16, 16, 16))
        y = block(x, True)
        assert y.shape == (1, 8, 8, 16)

    fn()  # type: ignore


def test_conv_transpose_block_call() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        block = nn.ConvTransposeBlock(size=16, kernel_shape=3, stride=2, dropout=0.1)
        x = jnp.ones((1, 8, 8, 16))
        y = block(x, True)
        assert y.shape == (1, 16, 16, 16)

    fn()  # type: ignore


def test_vae_encoder_call() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        encoder = nn.VAEEncoder(sizes=[16, 32, 64],
                                strides=[2, 1, 2],
                                out_size=32,
                                dropout=0.1)
        x = jnp.ones((1, 16, 16, 3))
        z = encoder(x, True)
        assert z.mean.shape == (1, 4, 4, 32)
        assert z.stddev.shape == (1, 4, 4, 32)

    fn()  # type: ignore


def test_vae_decoder_call() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        decoder = nn.VAEDecoder(sizes=[64, 32, 16],
                                strides=[2, 1, 2],
                                out_size=3,
                                dropout=0.1)
        x = jnp.ones((1, 2, 2, 128))
        y = decoder(x, True)
        assert y.shape == (1, 8, 8, 3)

    fn()  # type: ignore


def test_vae_encoder_decoder_call() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        encoder = nn.VAEEncoder(sizes=[16, 32, 64],
                                strides=[2, 2, 2],
                                out_size=32,
                                dropout=0.1)
        decoder = nn.VAEDecoder(sizes=[64, 32, 16],
                                strides=[2, 2, 2],
                                out_size=3,
                                dropout=0.1)
        x = jnp.ones((1, 16, 16, 3))
        z = encoder(x, True)
        x_hat = decoder(z.mean, True)
        assert z.mean.shape == (1, 2, 2, 32)
        assert z.stddev.shape == (1, 2, 2, 32)
        assert x_hat.shape == x.shape

    fn()  # type: ignore


def test_vae_call() -> None:

    @hk.testing.transform_and_run
    def fn() -> None:
        vae = nn.VAE(encoder_sizes=[16, 32, 64],
                     encoder_strides=[2, 2, 2],
                     decoder_sizes=[64, 32, 16],
                     decoder_strides=[2, 2, 2],
                     latent_size=32,
                     image_channels=3,
                     dropout=0.1)
        x = jnp.ones((1, 16, 16, 3))
        result = vae(x, True)
        assert result.mean.shape == (1, 2, 2, 32)
        assert result.stddev.shape == (1, 2, 2, 32)
        assert result.x_hat.shape == x.shape
        assert result.reconstruction_loss.shape == ()
        assert result.kl_loss.shape == ()

    fn()  # type: ignore
