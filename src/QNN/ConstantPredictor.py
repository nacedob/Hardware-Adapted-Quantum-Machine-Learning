"""
Hago este optimizador que sea lo que sea lo da al label que más tenga el training set, es decir, es la
opción de hacerlo "random" más favorable, o lo mejor a batir
"""

from icecream import ic
import pennylane.numpy as np


class ConstantPredictor:

    def __init__(self):
        self.trained = False
        self.constant = 0


    def train(self, train_set, label_train, test_set=None, label_test=None, silent:bool=True):

        train_len = len(label_train)
        one_len = np.count_nonzero(label_train)
        if one_len >= train_len/2:
            self.constant = 1

        if test_set is not None and label_test is not None:
            success = np.sum(label_test == self.constant)
            if not silent:
                print('[INFO] Evaluating model on test_set')
                print(f'[INFO] Accuracy: {success/len(test_set)}')

        self.trained = True

    def predict(self, set):
        return np.full(len(set), self.constant)

    def get_accuracy(self, set, labels):
        assert len(set) == len(labels)
        prediction = self.predict(set)
        return sum(prediction == labels)/len(set)

    def __str__(self):
        class_name = self.__class__.__name__
        return f'Constant predictor {class_name}.\nTrained: {"yes" if self.trained else "no"}'

