#!/usr/bin/env python3
'''The training loop and loss function. Also implements some auxiliary
functions such as automatic logging, etc.'''
from __future__ import annotations

import csv
import logging
import sys
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import count
from pathlib import Path
from typing import (Any, Dict, Iterator, List, Optional, Protocol, Tuple, Type,
                    TypeVar)

import click
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import optax
from chex import Array, ArrayTree, PRNGKey
from einops import rearrange
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # If the module is executed we need to add the parent module to the discoverable imports
    sys.path.append('.')

from minivae import common, data, nn

logger = logging.getLogger(common.NAME)


T = TypeVar('T')


class TrainingConfig(nn.ModelConfig, Protocol):
    '''Configuration for training.'''
    batch_size: int
    gradient_accumulation_steps: int  # Must divide batch_size
    use_half_precision: bool
    loss_scale_period: Optional[int]
    initial_loss_scale_log2: Optional[int]
    peak_learning_rate: float
    end_learning_rate: float
    warmup_steps: int
    total_steps: Optional[int]
    weight_decay: float

    @classmethod
    @abstractmethod
    def from_yaml(cls: Type[T], path: Path) -> T:
        raise NotImplementedError

    @abstractmethod
    def to_yaml(self: T, path: Path) -> T:
        raise NotImplementedError


@dataclass
class TelemetryData:
    '''Data to be logged during training.'''
    step: int
    epoch: int
    params: ArrayTree
    opt_state: optax.OptState
    loss_scale: jmp.LossScale
    config: TrainingConfig
    rngs: hk.PRNGSequence
    gradients: ArrayTree
    gradients_finite: bool
    loss: Array
    rec_loss: Array
    kl_loss: Array


@common.consistent_axes
def train(config: TrainingConfig,
          params: ArrayTree,
          opt_state: optax.OptState,
          dataloader: DataLoader,
          rngs: hk.PRNGSequence,
          loss_scale: Optional[jmp.LossScale] = None,
          step: int = 0,
          ) -> Iterator[TelemetryData]:
    '''Train the model, yielding telemetry data at each step.'''
    # Preparations
    policy = get_policy(config)
    loss_scale = get_loss_scale(config, step) if loss_scale is None else loss_scale
    train_step_jit = jax.pmap(partial(train_step, config=config, axis_name='device'),
                              axis_name='device',
                              donate_argnums=5)
    # Helper function for dealing with multiple devices
    device_count = jax.device_count()
    logger.info(f'Devices found: {device_count}.')
    # Broadcast components across devices
    params = broadcast_to_devices(params)
    opt_state = broadcast_to_devices(opt_state)
    loss_scale = broadcast_to_devices(loss_scale)
    # Training loop
    for epoch in count():
        for samples in dataloader:
            common.assert_shape(samples, 'B H W C')
            # Note that due to the JAX's asynchronous dispatch, the timing
            # information is not accurate within one step and should only be
            # considered across multiple steps.
            # See: https://jax.readthedocs.io/en/latest/async_dispatch.html
            samples = policy.cast_to_compute(samples)
            # Split samples and RNG between devices
            samples = rearrange(samples, '(d b) ... -> d b ...', d=device_count)
            rng = jax.random.split(next(rngs), num=device_count)
            params, opt_state, loss_scale, telemetry_dict = train_step_jit(
                samples, params, opt_state, loss_scale, rng)
            yield TelemetryData(
                step=step,
                epoch=epoch,
                params=get_from_first_device(params),
                opt_state=get_from_first_device(opt_state),
                loss_scale=get_from_first_device(loss_scale),
                config=config,
                rngs=rngs,
                gradients=get_from_first_device(telemetry_dict['gradients']),
                loss=jnp.mean(telemetry_dict['loss']),
                rec_loss=jnp.mean(telemetry_dict['rec_loss']),
                kl_loss=jnp.mean(telemetry_dict['kl_loss']),
                gradients_finite=telemetry_dict['gradients_finite'].all())
            step += 1
        logger.info(f'Epoch {epoch + 1:,} finished')


def train_step(samples: Array,
               params: ArrayTree,
               opt_state: optax.OptState,
               loss_scale: jmp.LossScale,
               rng: PRNGKey,
               *,
               config: TrainingConfig,
               axis_name: str,
               ) -> Tuple[ArrayTree,
                          optax.OptState,
                          jmp.LossScale,
                          Dict[str, Any]]:
    # Preparations
    common.assert_shape(samples, 'B H W C')
    loss_hk = hk.transform(partial(loss_fn, config=config))
    grad_fn = jax.grad(loss_hk.apply, has_aux=True)
    optimizer = get_optimizer(config)
    # Execution
    gradients, telemetry_dict = grad_fn(params, rng, samples)
    gradients = jax.lax.pmean(gradients, axis_name=axis_name)
    gradients = loss_scale.unscale(gradients)
    gradients_finite = jmp.all_finite(gradients)
    loss_scale = loss_scale.adjust(gradients_finite)
    updates, new_opt_state = optimizer.update(gradients, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    # Only actually update the params and opt_state if all gradients were finite
    opt_state, params = jmp.select_tree(
        gradients_finite,
        (new_opt_state, new_params),
        (opt_state, params))
    return (params,
            opt_state,
            loss_scale,
            dict(telemetry_dict,
                 gradients=gradients,
                 gradients_finite=gradients_finite))


def loss_fn(samples: Array,
            *,
            config: TrainingConfig,
            ) -> Tuple[Array, Dict[str, Any]]:
    model = nn.VAE.from_config(config)
    common.assert_shape(samples, 'B H W C',
                        B=config.batch_size,
                        H=config.shape[0],
                        W=config.shape[1],
                        C=config.shape[2])
    # Accumulate the loss
    samples_splits = rearrange(samples, '(o b) ... -> o b ...',
                               o=config.gradient_accumulation_steps)
    loss = jnp.zeros((), dtype=jnp.float32)
    rec_losses = []
    kl_losses = []
    for split in samples_splits:
        common.assert_shape(split, 'S H W C',
                            S=config.batch_size // config.gradient_accumulation_steps)
        output: nn.VAEOutput = model(split, is_training=True)
        rec_losses.append(output.reconstruction_loss)
        kl_losses.append(output.kl_loss)
    rec_loss = jnp.mean(jnp.asarray(rec_losses))
    kl_loss = jnp.mean(jnp.asarray(kl_losses))
    alpha = 0.9
    loss = alpha * rec_loss + (1 - alpha) * kl_loss
    return loss, dict(rec_loss=rec_loss,
                      kl_loss=kl_loss,
                      loss=loss)


def get_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    '''Get the optimizer with linear warmup and cosine decay.'''
    return optax.adamw(get_learning_rate_schedule(config),
                       weight_decay=config.weight_decay)


def get_policy(config: TrainingConfig) -> jmp.Policy:
    '''Get and set the policy for mixed precision training.'''
    # The VAE always uses full precision
    vae_policy = jmp.get_policy('params=f32,compute=f32,output=f32')
    hk.mixed_precision.set_policy(nn.VAE, vae_policy)
    # The Encoder and Decoder can use half precision internally
    half_policy = jmp.get_policy('params=f32,compute=f16,output=f32'
                                 if config.use_half_precision else
                                 'params=f32,compute=f32,output=f32')
    hk.mixed_precision.set_policy(nn.VAEEncoder, half_policy)
    hk.mixed_precision.set_policy(nn.VAEDecoder, half_policy)
    # LayerNorms use full precision internally and can use half precision for output
    ln_policy = jmp.get_policy('params=f32,compute=f32,output=f16'
                               if config.use_half_precision else
                               'params=f32,compute=f32,output=f32')
    hk.mixed_precision.set_policy(hk.LayerNorm, ln_policy)
    return vae_policy


def get_loss_scale(config: TrainingConfig,
                   step: int,
                   ) -> jmp.LossScale:
    '''Get the loss scale for mixed precision training.'''
    if config.use_half_precision:
        msg = 'initial_loss_scale_log2 must be set for mixed precision training.'
        assert config.initial_loss_scale_log2 is not None, msg
        msg = 'loss_scale_period must be set for mixed precision training.'
        assert config.loss_scale_period is not None, msg
        scale = jmp.DynamicLossScale(2. ** jnp.asarray(config.initial_loss_scale_log2),
                                     counter=jnp.asarray(step % config.loss_scale_period),
                                     period=config.loss_scale_period)
    else:
        scale = jmp.NoOpLossScale()
    return scale


def get_learning_rate_schedule(config: TrainingConfig) -> optax.Schedule:
    '''Get the learning rate schedule with linear warmup and optional cosine decay.'''
    if config.total_steps is not None:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.,
            peak_value=config.peak_learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.total_steps - config.warmup_steps,
            end_value=config.end_learning_rate,
        )
    else:
        schedules = [
            optax.linear_schedule(
                init_value=0.,
                end_value=config.peak_learning_rate,
                transition_steps=config.warmup_steps),
            optax.constant_schedule(config.peak_learning_rate),
        ]
        lr_schedule = optax.join_schedules(schedules, [config.warmup_steps])
    return lr_schedule


def get_optimizer_state(config: TrainingConfig,
                        params: ArrayTree,
                        ) -> optax.OptState:
    '''Get the optimizer state.'''
    optimizer = get_optimizer(config)
    opt_state = optimizer.init(params)
    opt_state_n = hk.data_structures.tree_size(opt_state)
    opt_state_mb = round(hk.data_structures.tree_bytes(opt_state) / 1e6, 2)
    logger.info(f'Optimizer state: {opt_state_n:,} ({opt_state_mb:.2f} MB)')
    return opt_state


def broadcast_to_devices(obj: T) -> T:
    device_count = jax.device_count()
    fn = lambda x: (jnp.broadcast_to(x, (device_count, *x.shape))
                    if isinstance(x, Array) else
                    x)
    return jax.tree_util.tree_map(fn, obj)


def get_from_first_device(obj: T) -> T:
    fn = lambda x: x[0] if isinstance(x, Array) else x
    return jax.tree_util.tree_map(fn, obj)


def concat_from_devices(obj: T) -> T:
    fn = lambda x: (rearrange(x, 'd b ... -> (d b) ...')
                    if isinstance(x, Array) else
                    x)
    return jax.tree_util.tree_map(fn, obj)


def autosave(telemetry_iter: Iterator[TelemetryData],
             frequency: int,
             path: Path,
             ) -> Iterator[TelemetryData]:
    '''Save the model parameters and optimizer state etc. at regular intervals.'''
    for telemetry in telemetry_iter:
        if not isinstance(telemetry.config, common.YamlConfig):
            raise ValueError('The config must be a YamlConfig to be saved.')
        if telemetry.step % frequency == 0:
            common.save_checkpoint(path,
                                   config=telemetry.config,
                                   params=telemetry.params,
                                   opt_state=telemetry.opt_state,
                                   rngs=telemetry.rngs,
                                   loss_scale=telemetry.loss_scale,
                                   step=telemetry.step)
        yield telemetry


def autolog(telemetry_iter: Iterator[TelemetryData],
            frequency: int,
            ) -> Iterator[TelemetryData]:
    '''Log the telemetry data at the specified frequency.'''
    loss_history = []
    rec_loss_history = []
    kl_loss_history = []
    for telemetry in telemetry_iter:
        loss_history.append(telemetry.loss)
        rec_loss_history.append(telemetry.rec_loss)
        kl_loss_history.append(telemetry.kl_loss)
        if telemetry.step % frequency == 0 and loss_history:
            mean_loss = jnp.mean(jnp.asarray(loss_history))
            mean_rec_loss = jnp.mean(jnp.asarray(rec_loss_history))
            mean_kl_loss = jnp.mean(jnp.asarray(kl_loss_history))
            logger.info(f'Step: {telemetry.step:,}'
                        f' | loss: {mean_loss:.4f}'
                        f' | rec: {mean_rec_loss:.4f}'
                        f' | kl: {mean_kl_loss:.4f}')
            loss_history.clear()
        yield telemetry


def log_to_csv(telemetry_iter: Iterator[TelemetryData],
               path: Path,
               ) -> Iterator[TelemetryData]:
    '''Log the telemetry data to a CSV file.'''
    lr_sched = None
    path.parent.mkdir(parents=True, exist_ok=True)
    did_exist = path.exists()
    if not did_exist:
        path.touch()
    with path.open('a') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'time', 'step', 'epoch', 'loss', 'rec_loss', 'kl_loss', 'learning_rate'])
        if not did_exist:
            writer.writeheader()
        for telemetry in telemetry_iter:
            if lr_sched is None:
                lr_sched = get_learning_rate_schedule(telemetry.config)
            writer.writerow(dict(time=datetime.now().isoformat(),
                                 step=telemetry.step,
                                 epoch=telemetry.epoch,
                                 loss=telemetry.loss,
                                 rec_loss=telemetry.rec_loss,
                                 kl_loss=telemetry.kl_loss,
                                 learning_rate=lr_sched(telemetry.step)))
            yield telemetry


class Config(common.YamlConfig):

    # Training config
    batch_size: int
    use_half_precision: bool
    loss_scale_period: Optional[int]
    initial_loss_scale_log2: Optional[int]
    gradient_accumulation_steps: int  # Must divide batch_size
    peak_learning_rate: float
    end_learning_rate: float
    warmup_steps: int
    total_steps: Optional[int]
    weight_decay: float

    # Model config
    encoder_sizes: List[int]
    encoder_strides: List[int]
    decoder_sizes: List[int]
    decoder_strides: List[int]
    latent_size: int
    dropout: float

    # Data config
    dataset_path: Path
    shape: Tuple[int, int, int]

    # DataLoader config
    num_workers: int


def get_cli() -> click.Group:
    '''Get the command line interface for this module.'''

    cli = common.get_cli_group('training')

    @cli.command('train')
    @click.option('--config-path', '-c', type=Path, default=None,
                  help='Path to the configuration file')
    @click.option('--load-from', '-l', type=Path, default=None,
                  help='Path to a checkpoint to resume training')
    @click.option('--save-path', '-o', type=Path, default=None,
                  help='Path to save checkpoints automatically')
    @click.option('--save-frequency', '-f', type=int, default=1000,
                  help='Frequency at which to save checkpoints automatically')
    @click.option('--log-frequency', type=int, default=10,
                  help='Frequency at which to log metrics automatically')
    @click.option('--csv-path', type=Path, default=None,
                  help='Path to save metrics in a CSV file')
    @click.option('--stop-at', type=int, default=None,
                  help='Stop training after this many steps')
    @click.option('--seed', type=int, default=None, help='Random seed')
    def cli_train(config_path: Optional[Path],
                  load_from: Optional[Path],
                  save_path: Optional[Path],
                  save_frequency: int,
                  log_frequency: int,
                  csv_path: Optional[Path],
                  stop_at: Optional[int],
                  seed: Optional[int],
                  ) -> None:
        '''Train a VAE.'''
        if config_path is None and load_from is None:
            raise ValueError('Either a configuration file or a checkpoint must be provided')
        if config_path is not None and load_from is not None:
            raise ValueError('Only one of configuration file or checkpoint must be provided')
        if config_path is not None:
            config = Config.from_yaml(config_path)
            rngs = common.get_rngs(seed)
            params = nn.VAE.get_params(config, next(rngs))
            opt_state = get_optimizer_state(config, params)
            step = 0
            loss_scale = None
        else:
            assert load_from is not None
            checkpoint = common.load_checkpoint(load_from, config_class=Config)
            config = checkpoint['config']
            rngs = checkpoint['rngs']
            params = checkpoint['params']
            opt_state = checkpoint['opt_state']
            step = checkpoint['step']
            loss_scale = checkpoint['loss_scale']
        dataloader = data.LMDBDataset.from_config(config).get_dataloader_from_config(config)
        telemetry_iter = train(config=config,
                               params=params,
                               opt_state=opt_state,
                               dataloader=dataloader,
                               rngs=rngs,
                               loss_scale=loss_scale,
                               step=step)
        if save_path is not None:
            telemetry_iter = autosave(telemetry_iter, save_frequency, save_path)
        if csv_path is not None:
            telemetry_iter = log_to_csv(telemetry_iter, csv_path)
        telemetry_iter = autolog(telemetry_iter, log_frequency)
        i = stop_at if stop_at is not None else -1
        while i != 0:
            next(telemetry_iter)
            i -= 1

    return cli


if __name__ == '__main__':
    get_cli()()
