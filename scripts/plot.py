#!/usr/bin/env python3
'''Utilities to plot the outputs of a training run'''
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd


@click.command()
@click.argument('path', type=Path)
def main(path: Path) -> None:
    df = pd.read_csv(path)
    df.plot(x='step', y=['loss', 'rec_loss', 'kl_loss'])
    plt.grid()
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
