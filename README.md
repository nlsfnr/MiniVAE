# MiniVAE

A Variational Auto-Encoder (VAE) implemented in Jax.

## Quick tour

The main files are

- [`./minivae/nn.py`](/minivae/nn.py) - The model and its components.
- [`./minivae/training.py`](/minivae/training.py) - The training loop and loss
  function.
- [`./minivae/inference.py`](/minivae/inference.py) - Methods to use pretrained
  models for inference.

## Details

The main point of MiniVAE was to learn about VAEs. The resulting model is
therefore quite simple, i.e. it is simply a stack of [de-]convolutional layers.

The posterior is approximated with a Gaussian and penalised with the
KL-divergence, the reconstruction loss is simply the MSE.

During training, you can add a flag to log to a CSV file and then use the
`plot.py` script to show the loss curve. For training and inference, both
`minivar/training.py` and `minivae/inference.py` are executable.
