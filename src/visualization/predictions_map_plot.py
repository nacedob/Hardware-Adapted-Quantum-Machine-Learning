from math import pi
import pandas as pd
import matplotlib.pyplot as plt
from src.QNN import GateQNN
from src.utils import print_in_blue
from src.visualization import dark_pink, dark_violet
import argparse
from src.experiments.config_exp import get_dataset, get_qnn
from src.visualization import visualize_predictions_2d
from src.utils import get_root_path, load_json_to_dict
from icecream import ic
from copy import deepcopy
import os
from ast import literal_eval
import optuna

root = get_root_path()

FONTSIZE = 25
BATCH_SIZE = 24
EPOCHS = 30

parser = argparse.ArgumentParser(description='Run the QNN experiment.')
parser.add_argument('--dataset', type=str, default='corners')
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--qubits', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--exp_id', type=int, default=0)
parser.add_argument('--load', type=str, default='False')
parser.add_argument('--trials', type=int, default=35)
parser.add_argument('--jobs', type=int, default=7)


args = parser.parse_args()

dataset = args.dataset
layers = args.layers
qubits = args.qubits
seed = args.seed
exp_id = args.exp_id
load = literal_eval(args.load)
jobs = args.jobs
trials = args.trials


################ LOAD DATASET ################
x, y, _, _ = get_dataset(dataset, 400, 10, 'jax', points_dimension=2, seed=seed, scale=1)

qnns = {}
accs = {}
loss = {}
models = ['gate', 'mixed', 'pulsed']

################ DEFINE TUNING SCORE ################
def score(trial) -> float:
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    qnn = get_qnn(model=model, n_layers=layers, n_qubits=qubits, seed=seed, realistic_gates=False)
    df_train = qnn.train(x, y, n_epochs=EPOCHS, batch_size=BATCH_SIZE, optimizer_parameters={'lr': lr}, save_stats=False, silent=True)
    train_loss = df_train.iloc[-1]['loss'].item(0)
    return train_loss

################ GET QNN AND ACC ################
if load:
    load_folder = f'{root}/data/results/final_experiment/{dataset}/'
    qnn_folder = f'{load_folder}/trained_qnn/{{model}}/exp_{exp_id}/qnn_q_{qubits}_l_{layers}_s_{seed}.pkl'
    for model in models:
        if model == 'gate':
            qnn = GateQNN.load_qnn(qnn_folder.format(model=model))
        else:
            qnn = GateQNN.load_qnn(qnn_folder.format(model=model))
        qnns[model] = qnn
        try:
            accs[model] = load_json_to_dict(qnn_folder.format(model=model).replace('pkl', 'json'))[
                'final_accuracy_train']
        except KeyError:
            df = pd.read_csv(f'{load_folder}/results_{exp_id}.csv')
            df = df[(df['n_qubits'] == qubits) & (df['n_layers'] == layers) & (df['seed'] == seed)]
            if model == 'gate':
                accs[model] = df['pulsed_acc_train'].values
            elif model == 'mixed':
                accs[model] = df['mixed_acc_train'].values
            elif model == 'pulsed':
                accs[model] = df['pulsed_acc_train'].values
            else:
                raise ValueError(f'Model {model} not recognized')
            exp_acc = qnn.get_accuracy(x, y)
else:
    for model in models:

        # Train
        if model == 'gate':
            qnn = get_qnn(model=model, n_layers=layers, n_qubits=qubits, seed=seed, realistic_gates=False)
            df = qnn.train(x, y, n_epochs=EPOCHS, batch_size=BATCH_SIZE, optimizer_parameters={'lr': 0.05},
                           save_stats=False, silent=True)

        else:
            study = optuna.create_study(direction="minimize", study_name=model)
            study.optimize(score, n_trials=trials, n_jobs=jobs)
            best_lr = study.best_params['lr']
            qnn = get_qnn(model=model, n_layers=layers, n_qubits=qubits, seed=seed, realistic_gates=False)
            df = qnn.train(x, y, n_epochs=EPOCHS, batch_size=BATCH_SIZE, optimizer_parameters={'lr': best_lr}, save_stats=False, silent=True)

        qnns[model] = qnn
        accs[model] = df.iloc[-1]['train_accuracy'].item(0)
        loss[model] = df.iloc[-1]['loss'].item(0)

################ PLOT ################
fig, ax = plt.subplots(1, 3, figsize=(25, 7))
for i, model in enumerate(models):
    ax[i] = visualize_predictions_2d(qnns[model], X=x, y=y, show=False, ax=ax[i], fontsize=FONTSIZE,
                                     mapcolor0=dark_pink, mapcolor1=dark_violet,
                                     pointcolor0='#ad6cad', pointcolor1=dark_violet, legend=False)
    ax[i].set_title(f'{model.capitalize()} - Accuracy: {accs[model]}', fontsize=FONTSIZE)

    ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize=FONTSIZE-5)
    ax[i].set_yticklabels(ax[i].get_yticklabels(), fontsize=FONTSIZE-5)


fig.suptitle(dataset.capitalize(), fontsize=FONTSIZE+3)
fig.tight_layout()
os.makedirs(f'{root}/data/final_plots/prediction_map', exist_ok=True)
path = f'{root}/data/final_plots/prediction_map/{dataset}_q_{qubits}_l_{layers}_s_{seed}.png'
fig.savefig(path)
print_in_blue(f'Figure saved in path:  {path}')
plt.close(fig)
