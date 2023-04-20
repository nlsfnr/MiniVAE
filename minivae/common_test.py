import numpy as np
import pytest

from minivae import common


def test_assert_shape() -> None:
    x = np.zeros((2, 3, 4))
    common.assert_shape(x, 'A B C')


def test_assert_shape_fail() -> None:
    x = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match='Expected axes'):
        common.assert_shape(x, 'A B C D')


def test_assert_shape_fail2() -> None:
    x = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match='Expected axis A'):
        common.assert_shape(x, 'A B C', A=3)


def test_assert_shape_fail3() -> None:
    x = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match='Expected axes'):
        common.assert_shape(x, 'A B')


def test_assert_shape_with_consistent_axes() -> None:

    @common.consistent_axes
    def fn(x: np.ndarray) -> None:
        common.assert_shape(x, 'A B C')
        y = x ** 2
        common.assert_shape(y, 'A B C')

    fn(np.zeros((2, 3, 4)))


def test_assert_shape_with_consistent_axes_fail() -> None:

    @common.consistent_axes
    def fn(x: np.ndarray) -> None:
        common.assert_shape(x, 'A B C')
        y = x ** 2
        common.assert_shape(y, 'A B C D')

    with pytest.raises(ValueError, match='Expected axes'):
        fn(np.zeros((2, 3, 4)))


def test_assert_shape_with_consistent_axes_fail2() -> None:

    @common.consistent_axes
    def fn(x: np.ndarray) -> None:
        common.assert_shape(x, 'A B C')
        y = np.concatenate([x, x], axis=1)
        common.assert_shape(y, 'A B C')

    with pytest.raises(ValueError, match='Expected axis B'):
        fn(np.zeros((3, 3, 4)))


def test_assert_shape_with_consistent_axes_nested() -> None:

    @common.consistent_axes
    def fn1(x: np.ndarray) -> None:
        common.assert_shape(x, 'A B C')
        y = x + fn2(x)
        common.assert_shape(y, 'A B C')

    @common.consistent_axes
    def fn2(x: np.ndarray) -> np.ndarray:
        # Reuse name A with a different value and introduce new labels X and Y
        common.assert_shape(x, 'X Y A')
        return x

    fn1(np.zeros((2, 3, 4)))
