import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from designer import NeuralNetworkDesigner
from differential_evolution import DifferentialEvolution, sphere_function
from utils import round_number

from activation_function import functions


class NeuralNetwork:
    def __init__(self, input_layer: int, hidden_layer: int, output_layer: int) -> None:
        self.n = input_layer
        self.m = output_layer
        self.h = hidden_layer

        self.wih = np.zeros((hidden_layer, input_layer + 1))
        self.who = np.zeros((output_layer, hidden_layer + 1))

        # Transfer Function (Activation Function)
        self.tfh = []
        self.tfo = []

        self.connected = []

    def forward(self, inputs):
        activation = np.array(inputs)

        layer_outputs = []
        for i in range(self.h):
            if self.connected[i]:
                weights = np.array(self.wih[i][1:])
                net_inputs = np.dot(activation, weights) + self.wih[i][-1]
                layer_outputs.append(functions[int(self.tfh[i])](net_inputs))
            else:
                layer_outputs.append(0)

        activation = np.array(layer_outputs)

        layer_outputs = []
        for i in range(self.m):
            if self.connected[i]:
                weights = np.array(self.who[i][1:])
                net_inputs = np.dot(activation, weights) + self.who[i][-1]
                layer_outputs.append(functions[int(self.tfo[i])](net_inputs))
            else:
                layer_outputs.append(0)

        return [layer_outputs]

    def set_net_configuration(self, neurons: list):
        for i in range(self.h):
            self.wih[i] = neurons[i][1:-1]
            self.tfh.append(neurons[i][-1])
            self.connected.append(neurons[i][0])

        for i in range(self.m):
            self.who[i] = neurons[i][1:-1]
            self.tfo.append(neurons[i][-1])
            self.connected.append(neurons[i][0])

    def get_neurons(self, parameters: list, architecture) -> list:
        if not parameters:
            parameters = [1] * len(architecture)

        indices = [i for i, x in enumerate(architecture) if x == "topology"]

        neurons = [
            parameters[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
        ]

        return self._process_neuron(neurons)

    def _process_neuron(self, neurons):
        for i in range(len(neurons)):
            neurons[i][0] = round_number(neurons[i][0])
            neurons[i][-1] = round_number(neurons[i][-1])

        return neurons


if __name__ == "__main__":
    designer = NeuralNetworkDesigner(4, 3)
    n, m, h = designer.n, designer.m, designer.h

    limits = designer.create_limits()
    neurons = designer.architecture

    de = DifferentialEvolution(pop_size=30, cr=0.9, f=0.5)
    pop = de.generate_population(bounds=limits)
    best = de.fit(fitness=sphere_function, max_iter=10)

    model = NeuralNetwork(input_layer=n, hidden_layer=h, output_layer=m)
    neurons = model.get_neurons(best[0], neurons)

    model.set_net_configuration(neurons)

    X, y = load_iris(return_X_y=True)

    predicted = []
    for item in X_train:
        outputs = model.forward(item)

        predicted.append(np.argmax(outputs))

    print(predicted)
