# Hardware-Adapted-Quantum-Machine-Learning
This repository explores Hardware-Adapted Quantum Machine Learning. It is part of my final msc thesis.


# Embedding-Quantum-Kernels-with-QNN

This code implements the paper by Pablo Rodriguez Grasa [Training embedding quantum kernels with data re-uploading quantum neural networks](https://arxiv.org/pdf/2401.04642).

## Abstract
Kernel methods play a crucial role in machine learning and the Embedding Quantum Kernels (EQKs), an extension to quantum systems, have shown very promising performance. However, choosing the right embedding for EQKs is challenging. We address this by proposing a p-qubit Quantum Neural Network (QNN) based on data re-uploading to identify the optimal q-qubit EQK for a task (p-to-q). This method requires constructing the kernel matrix only once, offering improved efficiency. In particular, we focus on two cases: n-to-n, where we propose a scalable approach to train an n-qubit QNN, and 1-to-n, demonstrating that the training of a single-qubit QNN can be leveraged to construct powerful EQKs.

## Code structure
The code is structured as follows:

### Quantum Neural Networks:
- [src/dataReuploading](src/dataReuploading) folder contains code containing the Quantum Neural Network part. 
  - [SingleQNN](src/dataReuploading/SingleQNN.py) models 1 qubit neural network.
  - [MultiQNN](src/dataReuploading/MultiQNN.py) models n qubit neural networks, being n arbitrary. In principle, taking  `n_qubits=1` should get the same results and object than the SingleQNN code.
  - [Constant](src/dataReuploading/Constant.py) is not actually a QNN. It takes the training set and seeks the most popular layer. Then, it maps every possible state in the Hilbert space to that specific label, with no computations being done. It is used as the *enemy* to beat, the minimum accuracy of the QNN and EQK to be achieved.

Any network in any QNN class has a `train` method as well as a method to save the python object containing the net (`save_qnn`) and the corresponding method to load it (`load_qnn`), the latter being static.

### Embedding Quantum Kernels:

- [src/EQK](src/EQK) folder contains code containing the Embedding Quantum Kernel part.
  - [EQK1](src/EQK/EQK1.py) code contains model the 1-1 architecture (more in Pablo's paper)
  - [EQK1](src/EQK/EQKn.py) code contains model the n-n architecture (more in Pablo's paper)

### Data sets

Some interesting data sets have been included in the within the class [sampler](Sampler). There are two classes: one modelling 2D data sets and another one implementing the 3D version of these same sets. 

One can try any other data set. Feel free to write your one methods in these classes.

Note that the methods in [visualization](Sampler/visualization) might be incorrect and raise error. I will be working in this soon.

### Tests

Almost every method and class in this code has their corresponding unitary tests that are located in [test folder](tests). They have been written with `unittest`. Note that some of the tests have been abandoned or removed. In principle, the classes are correctly implemented. Should you find any error, you can send me a message. I really appreciate collaboration.

## Example

For instance, for a MultiQNN:

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
