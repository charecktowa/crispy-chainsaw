import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from designer import NeuralNetworkDesigner
from differential_evolution import DifferentialEvolution
from network import NeuralNetwork


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def train(X, y, limits, n, m, h, neurons):
    evolution = DifferentialEvolution(pop_size=100, cr=0.9, f=0.5)
    pop = evolution.generate_population(bounds=limits)

    def fitness_function(individual):
        if individual is None:
            individual = pop

        model = NeuralNetwork(input_layer=n, hidden_layer=h, output_layer=m)

        config = model.get_neurons(individual, neurons)

        model.set_net_configuration(config)

        predicted = []
        for item in X:
            output = model.forward(item)
            predicted.append(np.argmax(output))

        return 1 - accuracy(y, predicted)

    best = evolution.fit(fitness=fitness_function, max_iter=100
    return best


def test(X, y, best, n, m, h, neurons):
    model = NeuralNetwork(input_layer=n, hidden_layer=h, output_layer=m)

    config = model.get_neurons(best, neurons)

    model.set_net_configuration(config)

    predicted = []
    for item in X:
        output = model.forward(item)
        predicted.append(np.argmax(output))

    return accuracy(y, predicted)


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        test_size=0.2,
        stratify=y,
        shuffle=True,
    )

    inp = X[0].shape
    oup = len(np.unique(y))

    designer = NeuralNetworkDesigner(input_dim=inp[0], output_dim=len(np.unique(y)))

    n, m, h = designer.n, designer.m, designer.h

    limits = designer.create_limits()
    neurons = designer.architecture

    x = train(
        X=X_train,
        y=y_train,
        limits=limits,
        neurons=neurons,
        n=n,
        m=m,
        h=h,
    )

    import matplotlib.pyplot as plt

    plt.plot(x[2], ".-")
    plt.xlabel("Improvement number")
    plt.ylabel("Fitness")
    plt.show()

    y = test(X_train, y_train, x[0], n, m, h, neurons)
    z = test(X_test, y_test, x[0], n, m, h, neurons)

    print(f"El mejor individuo fue: {x[0]} con fitnessde {x[1]}")

    print(f"El accuracy del mejor individuo en el conjunto de entrenamiento fue {y}")
    print(f"El accuracy del mejor individuo en el conjunto de prueba fue {z}")
