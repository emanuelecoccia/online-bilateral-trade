# Online Bilateral Trade

This repository contains the implementation of online algorithms for bilateral trade problems. It can be used to test algorithms in contextual and non-contextual online environments.

## Installation

To install this library, clone the repository and install the dependencies listed in the `requirements.txt` file.

```bash
# Clone the repository
git clone https://github.com/emanuelecoccia/online-bilateral-trade.git

# Navigate to the project directory
cd online-bilateral-trade

# Install the library
pip install .
```

Alternatively, you can install the library directly using pip:

```bash
pip install git+https://github.com/emanuelecoccia/online-bilateral-trade.git
```

## Usage

After installation, you can run experiments using the provided scripts or use the library in your own Python code to test various algorithms.

```python
from obt.environments.contextual import OrderBookEnvironment
from obt.learners.experts import ExploitNearestContext

# Create an instance of the bilateral trade environment
T = 1000
env = OrderBookEnvironment(T)

# Initialize and run your algorithm
L = 10
algo = ExploitNearestContext(T, env, L)
algo.run()
```

More details on how to configure the environment and algorithms can be found in the example scripts provided in the `notebooks/` directory.

## Citing This Work

If you use this library in your research or project, please cite it as follows:

```
@misc{emanuelecoccia2025,
  author = {Emanuele Coccia},
  title = {Online Bilateral Trade Testing Library},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/emanuelecoccia/online-bilateral-trade}}
}
```

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, feedback, or collaboration opportunities, feel free to contact [Emanuele Coccia] at [emanuele.coccia@studbocconi.it].
