# Hardware-Adapted-Quantum-Machine-Learning

**Author**: Ignacio Benito Acdo Blanco  
**Institution**: University Charles III of Madrid  
**Thesis Title**: *Quantum Hardware Adapted Machine Learning*

This repository implements the research from my master's thesis, *Hardware-Adapted Quantum Machine Learning*. The project introduces a pioneering framework for Pulsed Quantum Machine Learning (PQML), replacing conventional parameterized quantum gates with hardware-efficient parameterized quantum pulses. It also features an innovative encoding method using quantum pulse control to optimize model performance in the NISQ era.

## Features

- **PQML Framework**: Custom framework for training quantum neural networks using pulse programming.
- **Pulse-Based Encoding**: New encoding method tailored for hardware constraints.
- **Quantum Embedding Kernels**: Integration of pulse-trained quantum neural networks in classification tasks.
- **Benchmarking**: Tools for comparing PQML models with traditional gate-based quantum machine learning models.

## Code Structure

### Quantum Neural Networks

- **QNN Models**: Located in [src/QNN](src/QNN), includes:
  - **BaseQNN**: Abstract class for QNNs with methods for training, prediction, and accuracy calculation.
  - **GateQNN**: Implements QNN using traditional parameterized gates.
  - **PulsedQNN**: Implements QNN using pulsed quantum operations via Trotter-Suzuki approximation. Allows hardware-specific configurations via parameters like `encoding`.

### Pennypulse

This is an installable package that provides modified functions for Pennylane, enabling pulse simulations for transmon qubits. It includes functionality to define Trotter-Suzuki evolution for both single-qubit and two-qubit pulses. The package is a critical dependency used by the `PulsedQNN` class.

To run this project, ensure all requirements, including `Pennypulse`, are installed by executing:

```bash
python -m pip install -r ./requirements.txt
```

### Datasets

- **Samplers**: In [src/Sampler](src/Sampler):
  - **Sampler2D**: Generates synthetic 2D datasets (e.g., corners, sinus, circles, spirals).
  - **Sampler3D**: Generates synthetic 3D datasets (e.g., corners3d, sinus3d, shell, helix).
  - **MNISTSampler**: Loads and preprocesses MNIST Fashion and Handwritten datasets.
  - **RandomSampler**: Creates random datasets with varying difficulty levels.

### Experiments

- **Simulation Scripts**: In [src/experiments](src/experiments):
  - **final_experiment.py**: Trains GateQNN, MixedQNN, and PulsedQNN on a fixed dataset.
  - **main.py**: Runs datasets from the [Supplementary Materials document](Documents/Suplementary_Materials___Hardware_Adapted_QML.pdf), including new datasets introduced in the thesis.
  - **pulse_encoding.py**: Runs one encoding step for the pulsed and gate-based encodings and plots it on the Bloch Sphere.
### Tests

- **Unit Tests**: Located in [tests](tests), extensively covering methods and classes using `unittest`. Contributions and error reports are welcome for continuous improvement.


### Visualization
Some functions for visualizing the results of the experiments are located in [src/Visualization](src/Visualization). 

### Data
In the [data folder](data), one can find the images presented in the documents as well as some results for additional experiments.


## Example

To illustrate, here’s how to create and train a PulsedQNN using Brisbane hardware with the novel pulsed encoding:

```python
import matplotlib.pyplot as plt
from src.QNN import GateQNN, PulsedQNN
from src.Sampler import Sampler
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Generate synthetic data (example: circle dataset)
data, labels = Sampler().circle(n_points=200, seed=SEED)
data_test, labels_test = Sampler().circle(n_points=200, seed=SEED)

# Train QNNs
layers = 5
gateqnn = GateQNN(num_layers=layers, num_qubits=2, realistic_gates=False, seed=SEED)
pulsedqnn = PulsedQNN(num_layers=layers, num_qubits=2)

# Train and evaluate models
df_gate = gateqnn.train(data, labels, data_test, labels_test, silent=False)
df_pulsed = pulsedqnn.train(data, labels, data_test, labels_test, silent=False)

# Print and compare results
print('Training stats gate QNN:\n', df_gate, '\n\nTraining stats pulsed QNN:\n', df_pulsed)
```