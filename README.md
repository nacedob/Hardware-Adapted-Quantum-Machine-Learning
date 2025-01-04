# Hardware-Adapted-Quantum-Machine-Learning
This repository explores Hardware-Adapted Quantum Machine Learning. It is part of my final msc thesis.

It explores Embedding Quantum Kernels (EQK) with Quantum Neural Networks (QNN) using Data Reuploading. This model is based on a the paper by Pablo Rodriguez Grasa [Training embedding quantum kernels with data re-uploading quantum neural networks](https://arxiv.org/pdf/2401.04642).

The main purpose of this repository is to adapt this model to improve its actual implementation on current real quantum hardware in an efficient way. To do this, the main idea is to change parametrized gates by parametrized pulses. More details are found on the [final thesis document](todo.txt)

## Code structure
The code is structured as follows:

### Quantum Neural Networks:
- [src/QNN](src/QNN) folder contains code containing the Quantum Neural Network part, using Data Reuploading strategy. 
  - [BaseQNN](src/dataReuploading/SingleQNN.py): an abstract class modelling the idea of a QNN. The main methods are `train` to learn parameters, `predict` to make the prediction of a dataset and `get_accuracy`that computes the current accuracy of the model on a dataset with its correspoding expected labels. There is an abstract method `_base_circuit` to be defined by each of its children. This method assigns the quantum circuit to be used as core of the computations. 
  - [GateQNN](src/dataReuploading/GateQNN.py) models the gate (traditional) version of a QNN. The parametrized gates are arbitrary rotations, with 3 parameters each to be used.
  - [PulsedQNN](src/dataReuploading/PulsedQNN.py) models the pulsed version of a QNN. The pulses are modelled using Trotter-Suzuki approximation. There are multiple attributes that can be defined to specify the exact hardware implementation. TODO

### Embedding Quantum Kernels - package:
- [src/EQK](src/EQK) folder contains code containing the Embedding Quantum Kernel part.
  - [EQK1](src/EQK/EQKn.py) code contains model the n-n architecture (more in Pablo's paper)

### Datasets

Some interesting data sets have been included in the within the sampler package [Sampler](src/Sampler). There are four classes: 
- [Sampler2D](src/Sampler/Sampler2D.py): creates synthetic 2D datasets (including those appearing in the original reference - corners, sinus, circles and spirals).
- [Sampler3D](src/Sampler/Sampler3D.py): creates synthetic 3D datasets (including the three-dimensional version of those appearing in the original reference - corners3d, sinus3d, shell and helix).
- [MNISTSampler](src/Sampler/MNISTSampler.py): that loads and preprocesses MNIST Fashion and Handwritten datasets.
- [RandomSampler](src/Sampler/RandomSampler.py): that creates random datasets using `sklearn` package. There is a general method (`get_data`) where one can choose freely the paramters to create a random dataset and three preconfigures datasets (`random_easy`, `random_medium` and `random_hard`) that create random datasets with increasing difficulty.

### Visualization
TODO

### Experiments
TODO

### Tests

Almost every method and class in this code has their corresponding unitary tests that are located in [test folder](tests). They have been written with `unittest`. Note that some of the tests have been abandoned or removed. In principle, the classes are correctly implemented. Should you find any error, you can send me a message. I really appreciate collaboration.

## Example

For instance, create a PulsedQNN using brisbane hardware and the novel pulsed encoding: TODO (el codigo de abajo es a modificar tambien)

```python
import unittest
from src.dataReuploading import MultiQNN, SingleQNN
from src.Sampler import Sampler
from pennylane import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from warnings import warn
from time import time
from src.visualization import plot_predictions_2d
from src.utils import increase_dimensions

data, labels = Sampler().circle(n_points=200)
        data_test, labels_test = Sampler().circle(n_points=200)

        layers = 5

        multiqnn = MultiQNN(num_layers=layers, num_qubits=3)
        singleqnn = SingleQNN(num_layers=layers)

        df_multi = multiqnn.train(data, labels, data_test, labels_test, silent=False)
        df_single = singleqnn.train(data, labels, data_test, labels_test, silent=False)

        ic(df_single, df_multi)

        # Ensure that final loss does not differ in more than a 7.5%
        loss_single = df_single['loss'].values[-1]
        loss_multi = df_multi['loss'].values[-1]

        # Print predictions
        prediction_multi = multiqnn.predict(data_test)
        prediction_single = singleqnn.predict(data_test)
        fig, ax = plt.subplots(1,2)
        plot_predictions_2d(data_test, prediction_multi, labels_test, axis=ax[0],
                            title=f"multiqnn prediction - accuracy {multiqnn.get_accuracy(data, labels)}")
        plot_predictions_2d(data_test, prediction_single, labels_test, axis=ax[1],
                            title=f"singleqnn prediction - accuracy {multiqnn.get_accuracy(data, labels)}")
        plt.show()


```
