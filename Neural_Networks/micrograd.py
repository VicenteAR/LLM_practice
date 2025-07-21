import numpy as np
from graphviz import Digraph


class Value:

    def __init__(
        self,
        data,
        _children: set = (),
        _opp: str = "",
        label: str = "",
    ):
        self.data = data
        self._prev = set(_children)
        self._op = _opp
        self.grad = 0.0
        self.label = label
        self._backwards = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # Addition method. This creates a new Value object. Same with product
        # We wrap up constant values
        other = (
            other if isinstance(other, Value) else Value(other)
        )  # we ensure plus operation can be performed correctly
        out = Value(self.data + other.data, _children=(self, other), _opp="+")

        def _backwards():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backwards = _backwards
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = -1 * self
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = self + (-other)
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "Must be a real number"
        out = Value(self.data**n, _children=(self,), _opp=f"**{n}", label=self.label)

        def _backwards():
            self.grad += (n * self.data ** (n - 1)) * out.grad

        out._backwards = _backwards
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** (-1)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _opp="*")

        def _backwards():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backwards = _backwards
        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        out = (np.exp(2 * self.data) - 1.0) / (np.exp(2 * self.data) + 1.0)
        out = Value(out, _children=(self,), _opp="tanh")

        def _backwards():
            self.grad += (1 - (out.data) ** 2) * out.grad

        out._backwards = _backwards
        return out

    def exp(self):
        out = np.exp(self.data)
        out = Value(out, _children=(self,), _opp="exp")

        def _backwards():
            self.grad += out.data * out.grad

        out._backwards = _backwards
        return out

    def backwards(self):
        topo_order_list = []
        visited = set()

        def topo_order(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_order(child)
                topo_order_list.append(v)

        topo_order(self)

        self.grad = 1.0
        for node in reversed(topo_order_list):
            node._backwards()
        return topo_order_list


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(
        format=format, graph_attr={"rankdir": rankdir}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label="{ data %.4f | grad %.4f }" % (n.data, n.grad),
            shape="record",
        )
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


class Neuron:

    def __init__(self, n: int):
        assert isinstance(n, int)
        self.w = [
            Value(data=np.random.uniform(low=-1, high=1), _children=(), label="w")
            for _ in range(n)
        ]
        self.b = Value(data=np.random.uniform(low=0, high=1), _children=(), label="b")

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        assert isinstance(nin, int)
        assert isinstance(nout, int)
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        assert isinstance(nin, int)
        assert isinstance(nouts, list)
        step_dim = [nin] + nouts
        # We must create as many layers as the depth of the NN
        self.layers = [Layer(step_dim[i], step_dim[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
