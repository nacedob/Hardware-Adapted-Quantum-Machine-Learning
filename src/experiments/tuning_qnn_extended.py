"""
Este experimento tunea el learning rate de un modelo para un dataset concreto
"""
from math import pi
from src.utils import save_dict_to_json, print_in_green, print_in_blue, get_root_path
import optuna
import pandas as pd
from src.utils import pickle_extension
from icecream import ic
import argparse
import ast
import re
import os
from warnings import filterwarnings
from .config_exp import get_dataset, get_qnn

filterwarnings('ignore', category=FutureWarning)

SEED = 42

df_stats_partial = pd.DataFrame([], columns=['qnn_id', 'train_loss', 'acc_train', 'acc_test', 'score',
                                     'opt_params', 'n_epochs', 'seed'])
df_stats_partial.index.name = 'seed'

################ TUNING PARAMS #################################
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=float, help='Number of trials to tune hyperparameters',
                    required=False, default=75)
parser.add_argument('--n_qubits', type=int, help='Number of qubits', required=False, default=2)
parser.add_argument('--n_jobs', type=int, help='Number of simultaneous trials', required=False, default=5)
parser.add_argument('--n_seeds', type=int, help='Number of seeds', required=False, default=3)
parser.add_argument('--n_train', type=int, help='# points for train set', required=False, default=400)
parser.add_argument('--point_dimension', type=int, help='Dimension of dataset', required=False, default=3)
parser.add_argument('--n_test', type=int, help='# points for test set', required=False, default=200)
parser.add_argument('--optimizer', type=str, help='optimizer to be used', required=False, default='rms')
parser.add_argument('--model', type=str, help='model to train', required=False, default='pulsed')
parser.add_argument('--interface', type=str, help='interface', required=False, default='jax')
parser.add_argument('--allow_schedule', type=str, help='whether to allow schedules for optimizer',
                    required=False, default='False')
parser.add_argument('--realistic_gates', type=str, help='whether to use realistic gates for GateQNN',
                    required=False, default='True')
parser.add_argument('--tune_epochs', type=str, help='whether to tune epochs parameter',
                    required=False, default='False')
parser.add_argument('--tune_decay', type=str, help='whether to tune decay parameter',
                    required=False, default='False')
parser.add_argument('--tune_constant4amplitude', type=str, help='whether to tune constant4amplitude',
                    required=False, default='False')
parser.add_argument('--save_qnn', type=str, help='whether to save_qnn parameter',
                    required=False, default='False')
parser.add_argument('--fixed_epochs', type=int, help='number of epochs for training',
                    required=False, default=40)
parser.add_argument('--layers_min', type=int, help='#layers min to be compared', required=False, default=1)
parser.add_argument('--layers_max', type=int, help='#layers max to be compared', required=False, default=6)

args = parser.parse_args()
realistic_gates = ast.literal_eval(args.realistic_gates)
n_trials = args.trials
n_qubits = int(args.n_qubits)
n_jobs = int(args.n_jobs)
n_seeds = int(args.n_seeds)
n_epochs = int(args.fixed_epochs)
layers_min = int(args.layers_min)
layers_max = int(args.layers_max)
n_train = int(args.n_train)
n_test = int(args.n_test)
point_dimension = int(args.point_dimension)
optimizer = args.optimizer
interface = args.interface
model = args.model
tune_epochs = ast.literal_eval(args.tune_epochs)
allow_schedule = ast.literal_eval(args.allow_schedule)
tune_decay = ast.literal_eval(args.tune_decay)
tune_constant4amplitude = ast.literal_eval(args.tune_constant4amplitude)
save_qnn = ast.literal_eval(args.save_qnn)

if optimizer != 'rms':
    tune_decay = False


EXP_RESULTS_PATH = os.path.join(get_root_path(), f'data/results/tuning_extended/{model}')
QNN_SUBPATH = 'trained_qnn'
os.makedirs(EXP_RESULTS_PATH, exist_ok=True)

def get_qnn_path(exp_id, dataset, seed, n_layers):
    exppath = get_exp_path(exp_id, dataset, seed, n_layers)
    qnnpath = exppath.replace('results_', f'{QNN_SUBPATH}/results_')
    return qnnpath

################ TUNING GRID #################################
LR_MIN, LR_MAX = 0.0005, 0.08
DECAY_MIN, DECAY_MAX = 0.2, 0.95
if tune_epochs:
    EPOCH_MIN, EPOCH_MAX = 20, 80
else:
    EPOCH_MIN = EPOCH_MAX = n_epochs

if tune_constant4amplitude:
    CONSTANT4AMPLITUDE_MIN, CONSTANT4AMPLITUDE_MAX = 1e-10, 1e2
else:
    CONSTANT4AMPLITUDE_MIN = CONSTANT4AMPLITUDE_MAX = 1
BATCH_SIZE = 24
BETA1 = 0.9
BETA2 = 0.999
dataset_list = ['fashion']
# dataset_list = ['fashion', 'digits']


def suggest_optim_parameters(trial, try_boundaries: bool = True):
    """Suggest parameters"""
    # Decide whether to use fixed or dynamic learning rate
    if try_boundaries:
        use_fixed_lr = trial.suggest_categorical("use_fixed_lr", [True, False])
    else:
        use_fixed_lr = True
    if use_fixed_lr:
        lr = trial.suggest_float("lr", LR_MIN, LR_MAX, log=True)
        params = {"lr": lr}
    else:
        raise NotImplementedError
    return params


################ TUNING EXPERIMENT FUNCTIONS #################################
def train_and_evaluate(n_layers, n_epochs, batch_size, opt_params, seed, constant4amplitude):
    print(f'Starting trial with: {n_layers = }, {n_epochs = }, {batch_size = }, {opt_params = }, {constant4amplitude = }')

    # epochs_dict = {0: int(n_epochs / 3), -1: n_epochs}
    epochs_dict = n_epochs
    qnn = get_qnn(model, n_qubits, n_layers, realistic_gates, SEED, interface, constant4amplitude)

    df_train = qnn.train(data_points_train=train_set, data_labels_train=train_labels,
                         n_epochs=epochs_dict,
                         batch_size=batch_size,
                         optimizer=optimizer,
                         optimizer_parameters=opt_params,
                         silent=True,
                         save_stats=False,
                         )

    train_loss = df_train.iloc[-1]['loss'].item(0)
    final_acc_test = qnn.get_accuracy(test_set, test_labels)
    final_acc_train = qnn.get_accuracy(train_set, train_labels)
    ic(final_acc_test, final_acc_train, train_loss)
    score = train_loss

    # Save qnn
    qnn_path = get_qnn_path(0, dataset, seed, n_layers).replace(f'_seed_{seed}', '')   # el 0 este da igual luiego se quita
    os.makedirs(os.path.dirname(qnn_path), exist_ok=True)
    qnn_id = get_highest_id(os.path.dirname(qnn_path), 'qnn', f'{pickle_extension}') + 1
    qnn_path = qnn_path.replace('results_0', f'qnn_{qnn_id}').replace('csv', f'{pickle_extension}')
    if save_qnn:
        qnn.save_qnn(qnn_path)

        # save dict containing qnn parameters
        params_dict = {
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'n_epochs': epochs_dict,
            'batch_size': batch_size,
            'opt_params': opt_params,
            'final_accuracy_train': final_acc_train,
            'final_accuracy_test': final_acc_test,
            'train_loss': train_loss,
            'dataset': dataset,
            'point_dimension': point_dimension,
            'save_qnn': save_qnn,
            'optimizer': optimizer,
        }
        save_dict_to_json(params_dict, qnn_path.replace(f'{pickle_extension}', 'json'))

    # Save stats to global df
    stat_row = pd.DataFrame([{
        'qnn_id': qnn_id,
        'train_loss': train_loss,
        'acc_train': final_acc_train,
        'acc_test': final_acc_test,
        'score': score,
        'opt_params': opt_params,
        'n_epochs': n_epochs,
        'seed': seed,
    }])
    stat_row.index = [seed]
    stat_row.index.name = 'seed'
    global df_stats_partial
    df_stats_partial = pd.concat([df_stats_partial, stat_row], ignore_index=False)

    return score


def get_objective(n_layers, seed):
    def objective(trial):
        opt_params = suggest_optim_parameters(trial, try_boundaries=allow_schedule)  # TRY BOUNDARIES
        if EPOCH_MIN == EPOCH_MAX:
            epochs = n_epochs
        else:
            epochs = trial.suggest_int("epochs", EPOCH_MIN, EPOCH_MAX, log=False)

        if tune_constant4amplitude:
            constant4amplitude = trial.suggest_float("constant4amplitude",
                                                     CONSTANT4AMPLITUDE_MIN, CONSTANT4AMPLITUDE_MAX,
                                                     log=True)
        else:
            constant4amplitude = 1
        score = train_and_evaluate(n_layers, epochs, BATCH_SIZE, opt_params, seed, constant4amplitude)
        return score

    return objective


################ SAVE RESULTS #################################
def get_highest_id(folder_path, pattern: str = 'results', extension: str = 'csv'):
    pattern = re.compile(fr"{pattern}_(\d+)\.{extension}")  # Regex to match 'results_id.csv'
    highest_id = -1

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            file_id = int(match.group(1))
            highest_id = max(highest_id, file_id)

    return highest_id


def get_exp_path(exp_id: int, dataset, seed, n_layers):
    path = f'{EXP_RESULTS_PATH}/{dataset}/layers_{n_layers}/results_{exp_id}_seed_{seed}.csv'
    return path


def save_results(exp_id: int, study, df_stats: pd.DataFrame, config_dict: dict, n_layers, dataset, seed):
    path = get_exp_path(exp_id, dataset, seed, n_layers)

    # Save results
    results = [{"trial_number": trial.number,
                "params": trial.params,
                "cost": trial.value}
               for trial in study.trials]
    df = pd.DataFrame(results).sort_values(by="cost", ascending=False)
    df.to_csv(path, index=False)

    # Save stats
    path_stats = os.path.join(os.path.dirname(path), f'stats_{exp_id}.csv')
    df_stats.sort_values('score', ascending=False)
    df_stats.to_csv(path_stats, index=True)

    # Save experiment configurations
    path_config = os.path.join(EXP_RESULTS_PATH, f'config_{exp_id}.json')
    save_dict_to_json(config_dict, path_config)


################ MAIN #################################
if __name__ == '__main__':

    n_experiments = (layers_max - layers_min + 1) * len(dataset_list) * n_seeds * n_trials
    print_in_blue(f'Number of experiments to be performed {n_experiments}')

    for n_layers in range(layers_min, layers_max + 1):
        for dataset in dataset_list:
            for seed in range(n_seeds):
                objective = get_objective(n_layers, seed)
                path = get_exp_path(0, dataset, seed, n_layers)
                folder = os.path.dirname(path)
                os.makedirs(folder, exist_ok=True)

                print_in_green(f'[START PARTIAL] Tuning experiment for layers: {n_layers}, dataset: {dataset}, seed: {seed} ---started')
                train_set, train_labels, test_set, test_labels = get_dataset(dataset,
                                                                             n_train,
                                                                             n_test,
                                                                             interface,
                                                                             points_dimension=point_dimension,
                                                                             seed=seed, scale=pi)
                study = optuna.create_study(direction="minimize", study_name='Tuning Experiment')
                study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

                result_dic = study.best_params
                exp_id = get_highest_id(folder) + 1
                result_cost = study.best_value
                exp_dict = {
                    "model": model,
                    "n_trials": n_trials,
                    "n_jobs": n_jobs,
                    "n_train": n_train,
                    "n_test": n_test,
                    "dataset": dataset,
                    "LR_MIN": LR_MIN,
                    "LR_MAX": LR_MAX,
                    "layers_min": layers_min,
                    "layers_max": layers_max,
                    "n_qubits": n_qubits,
                    "BETA1": BETA1,
                    "BETA2": BETA2,
                    "tune_epochs": tune_epochs,
                    "tune_decay": tune_decay,
                    "tune_constant4amplitude": tune_constant4amplitude,
                    "CONSTANT4AMPLITUDE_MIN": CONSTANT4AMPLITUDE_MIN,
                    "CONSTANT4AMPLITUDE_MAX": CONSTANT4AMPLITUDE_MAX,
                    "n_epochs": n_epochs,
                    "epochs_min": EPOCH_MIN,
                    "epochs_max": EPOCH_MAX,
                    "optimizer": optimizer,
                    'save_qnn': save_qnn,
                }

                save_results(exp_id, study, df_stats_partial, exp_dict, n_layers, dataset, seed)

                print_in_green(f'[PARTIAL END] Tuning experiment for layers: {n_layers}, dataset: {dataset}, seed: {seed} ---finished')
                print(f'              Best Parameters: {result_dic},\n   Best Cost: {result_cost}')
                print('------')
    print_in_green('\n\n[END] Tuning experiment finished!')
