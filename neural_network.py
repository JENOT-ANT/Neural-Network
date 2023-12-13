#      [] - [] - []
#      [] - [] - [] \ []
# [] / [] - [] - [] - []
# [] - [] - [] - [] - []
# [] \ [] - [] - [] - []
#      [] - [] - [] / []
#      [] - [] - []

from random import normalvariate as rand
from math import exp



def relu(inputs: list[float]) -> list[float]:
    fixed: list[float] = []
    
    for value in inputs:
        fixed.append(max(0, value))
    
    return fixed

def soft_max(outputs: list[float]) -> list[float]:
    max_value: float = max(outputs)
    fixed: list[float] = []
    sum: float = 0

    for value in outputs:
        _ = exp(value - max_value)
        fixed.append(_)
        sum += _

    for i in range(len(fixed)):
        fixed[i] = fixed[i] / sum

    return fixed


class Neuron:
    weights: list[float]
    bias: float

    _test_weights: list[float]
    _test_bias: float


    def __init__(self, inputs_amount: int):
        self.bias = 0
        self.weights = [0 for _ in range(inputs_amount)]
        self._test_weights = [0 for _ in range(inputs_amount)]
        self._test_bias = 0

    def apply(self):
        self.weights = [value for value in self._test_weights]
        self.bias = self._test_bias

    def randomize(self, delta: float):
        self._test_bias = self.bias + delta * rand()

        for i in range(len(self.weights)):
            self._test_weights[i] = self.weights[i] + delta * rand()
    
    def test_forward(self, inputs: list[float]):
        output: float = 0

        if len(self._test_weights) != len(inputs):
            raise Exception(f'Inputs do not match weights. {self}')

        for i in range(len(inputs)):
            output += inputs[i] * self._test_weights[i]
        
        return output + self._test_bias

    def forward(self, inputs: list[float]):
        output: float = 0

        if len(self.weights) != len(inputs):
            raise Exception(f'Inputs do not match weights. {self}')

        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]
        
        return output + self.bias

class Layer:
    neurons: list[Neuron]

    def __init__(self, inputs_amount: int, neurons_amount):
        self.neurons = []

        for i in range(neurons_amount):
            self.neurons.append(Neuron(inputs_amount))
    
    def apply(self):
        for neuron in self.neurons:
            neuron.apply()

    def randomize(self, delta: float):
        for neuron in self.neurons:
            neuron.randomize(delta)

    def test_forward(self, inputs: list[float]) -> list[float]:
        outputs: list = []

        for neuron in self.neurons:
            outputs.append(neuron.test_forward(inputs))
        
        return outputs

    def forward(self, inputs: list[float]) -> list[float]:
        outputs: list = []

        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        
        return outputs


class Network:
    layers: list[Layer]

    def __init__(self, layers_amount: int, layers_size: int, inputs_amount: int, outputs_amount: int):
        self.layers = []
        
        # Create network:
        # First Layer
        self.layers.append(Layer(inputs_amount, layers_size))

        # Hidden Layer
        for i in range(layers_amount - 1):
            self.layers.append(Layer(layers_size, layers_size))
        
        # Output Layer
        self.layers.append(Layer(layers_size, outputs_amount))

    def apply(self):
        for layer in self.layers:
            layer.apply()

    def randomize(self, delta: float, layer_index: int=-1):
        if layer_index == -1:
            for layer in self.layers:
                layer.randomize(delta)
        else:
            self.layers[layer_index].randomize(delta)

    def test_compute(self, inputs: list[float], activation=relu, output_activation=soft_max) -> list[float]:
        inputs = self.layers[0].test_forward(inputs)
        
        for i in range(1, len(self.layers)):
            inputs = self.layers[i].test_forward(activation(inputs))

        return output_activation(inputs)

    def compute(self, inputs: list[float], activation=relu, output_activation=soft_max) -> list[float]:
        inputs = self.layers[0].forward(inputs)
        
        for i in range(1, len(self.layers)):
            inputs = self.layers[i].forward(activation(inputs))

        return output_activation(inputs)
