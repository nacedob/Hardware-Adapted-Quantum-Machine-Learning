import numpy as np
from src.utils import get_root_path
from icecream import ic
from src.Sampler import MNISTSampler, Sampler
from src.QNN import GateQNN
from src.QNN.OriginalQNN import MultiQNN
from time import time
import pandas as pd
import optuna
import os
from src.utils import get_root_path


x_train, y_train = Sampler.circle(n_points=400)
x_test, y_test = Sampler.circle(n_points=200)

nq = 1
nl = 3
EPOCHS = 8
BATCH_SIZE = 24
LR = 0.1

ic('GateQNN')
qnn = GateQNN(num_qubits=nq, num_layers=nl)
t_start = time()
df = qnn.train(x_train, y_train, x_test, y_test, n_epochs=EPOCHS, batch_size=BATCH_SIZE,
               optimizer_parameters={'lr': LR},
               silent=False)
t_end = time() - t_start
df['Training_time'] = t_end * np.ones(len(df))
ic(t_end)

ic('Original')
orqnn = MultiQNN(num_qubits=nq, num_layers=nl)
t_start = time()
ordf = orqnn.train(x_train, y_train, x_test, y_test, n_epochs=EPOCHS, batch_size=BATCH_SIZE,
                   optimizer_parameters={'lr': LR},
                   silent=False)
t_end = time() - t_start
ordf['Training_time'] = t_end * np.ones(len(ordf))
ic(t_end)

ic(df, ordf)
combined_df = pd.concat([df.add_suffix('NewQNN'), ordf.add_suffix('OriginalQNN')], axis=1)
ic(combined_df)


root = get_root_path('Hardware-Adapted-Quantum-Machine-Learning')
path = os.path.join(root, 'data/results/old/original_vs_new_qnn.csv')
os.makedirs(os.path.dirname(path), exist_ok=True)
combined_df.to_csv(path)
print('Results successfully saved!')