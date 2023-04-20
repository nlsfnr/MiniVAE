'''Common functionality shared across modules.'''
import json
import logging
import pickle
import random
import sys
from functools import wraps
from pathlib import Path
from typing import (Any, Callable, Dict, List, Optional, Type, TypeVar, Union,
                    cast)

import chex
import click
import haiku as hk
import jax
import jmp
import numpy as np
import optax
import pydantic
import yaml
from PIL import Image

NAME = 'MiniVAE'

logger = logging.getLogger(NAME)


def set_debug(debug: bool) -> None:
    jax.config.update('jax_debug_nans', debug)
    jax.config.update('jax_debug_infs', debug)
    jax.config.update('jax_disable_jit', debug)
    if debug:
        logger.warning('Running in debug mode')


def get_rngs(seed: Optional[Union[hk.PRNGSequence, int]] = None,
             ) -> hk.PRNGSequence:
    '''Get a PRNG sequence from an int or an existing PRNG sequence.'''
    if isinstance(seed, hk.PRNGSequence):
        return seed
    seed = (random.randint(0, 2**32 - 1)
            if seed is None else
            seed)
    logger.info(f'Using seed {seed}')
    return hk.PRNGSequence(seed)


T = TypeVar('T', bound='YamlConfig')


class YamlConfig(pydantic.BaseModel):

    @classmethod
    def from_yaml(cls, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self: T, path: Path) -> T:
        with open(path, "w") as f:
            # Use self.json() instead of self.dict() to avoid having to catet
            # to edge cases such as serializing Paths.
            yaml.dump(json.loads(self.json()), f)
        return self


def save_checkpoint(path: Path,
                    config: YamlConfig,
                    params: chex.ArrayTree,
                    opt_state: optax.OptState,
                    rngs: hk.PRNGSequence,
                    loss_scale: jmp.LossScale,
                    step: int,
                    ) -> None:
    '''Save the checkpoint to a directory.'''
    path.mkdir(parents=True, exist_ok=True)
    # Save the configuration
    config.to_yaml(path / 'config.yaml')
    # Save the parameters
    with open(path / 'params.pkl', 'wb') as f:
        pickle.dump(params, f)
    # Save the optimizer state
    with open(path / 'opt_state.pkl', 'wb') as f:
        pickle.dump(opt_state, f)
    # Save the step as a yaml file
    with open(path / 'other.yaml', 'w') as f:
        yaml.dump(dict(step=step), f)
    # Save the RNGs
    with open(path / 'rngs.pkl', 'wb') as f:
        pickle.dump(rngs, f)
    # Save the loss scale
    with open(path / 'loss_scale.pkl', 'wb') as f:
        pickle.dump(loss_scale, f)
    logger.info(f'Saved checkpoint to {path} at step {step:,}.')


def load_checkpoint(path: Path,
                    config_class: Type[YamlConfig],
                    ) -> Dict[str, Any]:
    '''Load the checkpoint from a directory.'''
    config = config_class.from_yaml(path / 'config.yaml')
    # Load the parameters
    with open(path / 'params.pkl', 'rb') as f:
        params = pickle.load(f)
    # Load the optimizer state
    with open(path / 'opt_state.pkl', 'rb') as f:
        opt_state = pickle.load(f)
    # Load the step from the yaml file
    with open(path / 'other.yaml', 'r') as f:
        other = yaml.load(f, Loader=yaml.FullLoader)
    step = other['step']
    # Load the RNGs
    with open(path / 'rngs.pkl', 'rb') as f:
        rngs_internal_state = pickle.load(f).internal_state
    rngs = hk.PRNGSequence(0)
    rngs.replace_internal_state(rngs_internal_state)
    # Load the loss scale
    with open(path / 'loss_scale.pkl', 'rb') as f:
        loss_scale = pickle.load(f)
    return dict(config=config,
                params=params,
                opt_state=opt_state,
                rngs=rngs,
                step=step,
                loss_scale=loss_scale)


def setup_logging(log_level: str,
                  log_to_stdout: bool,
                  logfile: Optional[Path],
                  ) -> None:
    handlers: List[logging.Handler] = []
    handlers.append(logging.StreamHandler(sys.stdout if log_to_stdout else sys.stderr))
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=log_level,
                        format='[%(asctime)s|%(name)s|%(levelname)s] %(message)s',
                        handlers=handlers)


def get_cli_group(name: str) -> click.Group:
    '''Get a click group with a common set of options.'''
    full_name = f'{NAME} - {name}'

    @click.group(full_name)
    @click.option('--log-level', default='INFO', help='Log level')
    @click.option('--log-to-stdout', is_flag=True, help='Log to stdout instead of stderr')
    @click.option('--logfile', type=Path, default=Path('./logs.log'), help='Log file')
    @click.option('--debug', '-d', is_flag=True, help='Debug mode')
    def cli(log_level: str,
            log_to_stdout: bool,
            logfile: Optional[Path],
            debug: bool,
            ) -> None:
        '''MiniGPT, a GPT-like language model'''
        setup_logging(log_level, log_to_stdout, logfile)
        logger.info(f'Starting {full_name}')
        set_debug(debug)

    return cli


ArrayT = TypeVar('ArrayT', bound=Union[np.ndarray, chex.Array])
CallableT = TypeVar('CallableT', bound=Callable)
_AXES_STACK: List[Dict[str, int]] = []


def assert_shape(x: ArrayT, names: str, **values: int) -> ArrayT:
    '''Assert that the shape of x has the given names and values.

    This can be used in conjunction with `consistent_axes` to ensure that axes
    with the same name are consistent within one function call. Axes with a
    name starting with '_' will not be stored.

    Example:

        @consistent_axes
        def f(x: Array) -> Array:
            assert_shape(x, 'B N C', B=2)
            y = x ** 2
            assert_shape(y, 'B N C')

        @consistent_axes
        def f(x: Array) -> Array:
            assert_shape(x, 'B N C', B=2)
            y = jnp.concatenate([x, x], axis=1)
            assert_shape(y, 'B N C')  # Raises an error
    '''
    global _AXES_STACK
    # Parse axis names
    axes = [axis for axis in names.split() if axis]
    # Check rank
    if len(x.shape) != len(axes):
        raise ValueError(f'Expected axes {" ".join(axes)}, '
                         f'got {" ".join(map(str, x.shape))}')
    # Check against explicit values
    for axis, name in enumerate(axes):
        if name in values and x.shape[axis] != values[name]:
            raise ValueError(f'Expected axis {name} to be {values[name]}, '
                             f'got {x.shape[axis]}')
    # Check against consistent axis values
    if _AXES_STACK:
        for axis, name in enumerate(axes):
            expected = _AXES_STACK[-1].get(name, None)
            actual = x.shape[axis]
            if expected is not None and actual != expected:
                raise ValueError(f'Expected axis {name} to be {expected}, got {actual}')
            if name.startswith('_'):
                continue
            _AXES_STACK[-1][name] = actual
    return x


def consistent_axes(fn: CallableT,
                    inherit: str = '',
                    ) -> CallableT:

    @wraps(fn)
    def wrapped(*args, **kwargs) -> Any:
        global _AXES_STACK
        axes = dict()
        if inherit and _AXES_STACK:
            names = [axis for axis in inherit.split() if axis]
            parent = _AXES_STACK[-1]
            axes.update({name: parent[name] for name in names if name in parent})
        _AXES_STACK.append(axes)
        try:
            return fn(*args, **kwargs)
        finally:
            _AXES_STACK.pop()

    return cast(CallableT, wrapped)


def to_pil_image(src: Union[np.ndarray, chex.Array],
                 show: bool = False,
                 save: Optional[Path] = None,
                 ) -> Image.Image:
    '''Convert an array intto a PIL image, optionally showing or saving it.'''
    x = np.clip((np.asarray(src) * 127.5 + 127.5), 0, 255).astype(np.uint8)
    # Remove batch dimension
    if len(x.shape) == 4:
        if x.shape[0] == 1:
            x = x[0]
        else:
            raise ValueError(f'Expected a single image, got {x.shape}')
    # Grayscale
    if len(x.shape) == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if len(x.shape) == 2:
        image = Image.fromarray(x, 'L')
    # RGB
    else:
        channels = x.shape[-1]
        if channels != 3:
            raise ValueError(f'Expected 1 or 3 channels, got {channels}')
        image = Image.fromarray(x, 'RGB')
    # Save or show it
    if save is not None:
        image.save(save)
    if show:
        image.show()
    return image
