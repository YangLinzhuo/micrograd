"""
Autograd engine implementing reverse-mode autodifferentiation, aka backpropagation.

Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
"""

# https://stackoverflow.com/questions/35701624/pylint-w0212-protected-access
# pylint: disable=protected-access

from collections.abc import Callable
from typing import Self


class Scalar:
    """Stores a single scalar value and its gradient"""

    def __init__(
        self, data: float, children: tuple[Self, ...] = (), op: str = ""
    ) -> None:
        self.data: float = data #if isinstance(data, float) else data.data
        self.grad: float = 0

        # Internal variables used for autograd graph construction
        self._backward: Callable = lambda: None
        self._prev: set[Self] = set(children)
        self._op = (
            op  # The operation that produced this node, for graphviz / debugging / etc
        )

    def __add__(self, other: Self | float) -> Self:
        self_type: type[Self] = type(self)
        _other: Self = self_type(other) if isinstance(other, float) else other
        out: Self = self_type(self.data + _other.data, (self, _other), "+")

        def _backward() -> None:
            self.grad += out.grad  # d(out)/d(self) = 1
            _other.grad += out.grad  # d(out)/d(other) = 1

        out._backward = _backward

        return out

    def __sub__(self, other: Self | float) -> Self:
        self_type: type[Self] = type(self)
        _other: Self = self_type(other) if isinstance(other, float) else other
        out: Self = self_type(self.data - _other.data, (self, _other), "-")

        def _backward() -> None:
            self.grad += out.grad  # d(out)/d(self) = 1
            _other.grad -= out.grad  # d(out)/d(other) = -1

        out._backward = _backward

        return out

    def __mul__(self, other: Self | float) -> Self:
        self_type: type[Self] = type(self)
        _other: Self = self_type(other) if isinstance(other, float) else other
        out = self_type(self.data * _other.data, (self, _other), "*")

        def _backward() -> None:
            self.grad += out.grad * _other.data  # d(out)/d(self) = other
            _other.grad += out.grad * self.data  # d(out)/d(other) = self

        out._backward = _backward

        return out

    def __truediv__(self, other: Self | float) -> Self:
        self_type: type[Self] = type(self)
        _other: Self = self_type(other) if isinstance(other, float) else other
        out = self_type(self.data / _other.data, (self, _other), "/")

        def _backward() -> None:
            self.grad += out.grad / _other.data  # d(out)/d(self) = 1/other
            # d(out)/d(other) = -self/(other*other)
            _other.grad += out.grad * (-self.data / (_other.data * _other.data))

        out._backward = _backward

        return out

    def relu(self) -> Self:
        """Compute ReLU"""
        self_type: type[Self] = type(self)
        out = self_type(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        """Compute gradients through backpropagation"""

        # Topological order all the children in the graph
        topo: Vector = []
        visited: set[Scalar] = set()

        def build_topo(node: Scalar) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other: Self | float) -> Self:
        return self.__add__(other)

    def __rsub__(self, other: Self | float) -> Self:
        self_type: type[Self] = type(self)
        _other: Self = self_type(other) if isinstance(other, float) else other
        return _other.__sub__(self)

    def __rmul__(self, other: Self | float) -> Self:
        return self.__mul__(other)

    def __rtruediv__(self, other: Self | float) -> Self:
        self_type: type[Self] = type(self)
        _other: Self = self_type(other) if isinstance(other, float) else other
        return _other.__truediv__(self)

    def __repr__(self) -> str:
        return f"Scalar(data={self.data}, grad={self.grad})"


Vector = list[Scalar]
