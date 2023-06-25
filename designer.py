from differential_evolution import DifferentialEvolution, sphere_function

from utils import round_number


class NeuralNetworkDesigner:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.n = input_dim
        self.m = output_dim
        self.q = round_number(self._calculate_q())
        self.h = round_number(self._calculate_h())
        self.dim = self._calculate_dim()

        self.limits = {
            "topology": (0, 1),
            "weight": (-4, 4),
            "bias": (-2, 2),
            "af": (0, 4),
        }

    def create_limits(self):
        network = self._generate_network()

        return [self.limits[element] for element in network]

    def _generate_network(self) -> list[str]:
        network = []

        # Input - Hidden
        for _ in range(self.h):
            network.extend(["topology"] + ["weight"] * self.n + ["bias", "af"])

        # Hidden - Output
        for _ in range(self.m):
            network.extend(["topology"] + ["weight"] * self.h + ["bias", "af"])

        self.architecture = network
        return network

    def _calculate_q(self) -> float:
        return self.n + self.m + ((self.n + self.m) / 2)

    def _calculate_h(self) -> int:
        return self.q - (self.n + self.m)

    def _calculate_dim(self) -> int:
        return (self.h * (self.n + 3)) + (self.m * (self.h + 3))


if __name__ == "__main__":
    designer = NeuralNetworkDesigner(4, 3)
    print(
        f"Input neurons: {designer.n}, Hidden neurons: {designer.h}, output neurons: {designer.m}"
    )
    print(
        f"Total neurons: {designer.q}, Total elements for vector (DE): {designer.dim}"
    )

    network_limits = designer.create_limits()

    de = DifferentialEvolution(pop_size=30, cr=0.9, f=0.5)
    pop = de.generate_population(bounds=network_limits)
    x = de.fit(fitness=sphere_function, max_iter=200)

    print(x[0], len(x[0]))

    import matplotlib.pyplot as plt

    plt.plot(x[2], ".-")
    plt.xlabel("Improvement number")
    plt.ylabel("Fitness")
    plt.show()
