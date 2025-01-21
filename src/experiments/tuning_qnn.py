"""
Este experimento tunea el learning rate de un modelo para un dataset concreto
"""
from math import pi
import matplotlib.pyplot as plt
from src.utils import save_dict_to_json, get_root_path
import optuna
import pandas as pd
from icecream import ic
import argparse
import ast
import re
from src.visualization import visualize_predictions_2d
import os
from warnings import filterwarnings
from .config_exp import get_dataset, get_qnn
from src.utils import pickle_extension

filterwarnings('ignore', category=FutureWarning)

SEED = 42

df_stats_partial = pd.DataFrame([], columns=['qnn_id', 'train_loss', 'acc_train', 'acc_test', 'score'])

################ TUNING PARAMS #################################
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=float, help='Number of trials to tune hyperparameters',
                    required=False, default=25)
parser.add_argument('--n_qubits', type=int, help='Number of qubits', required=False, default=1)
parser.add_argument('--n_layers', type=int, help='Number of layers', required=False, default=4)
parser.add_argument('--epochs', type=int, help='Number of epochs', required=False, default=20)
parser.add_argument('--n_jobs', type=int, help='Number of simultaneous trials', required=False, default=4)
parser.add_argument('--n_train', type=int, help='# points for train set', required=False, default=500)
parser.add_argument('--point_dimension', type=int, help='Dimension of dataset', required=False, default=3)
parser.add_argument('--n_test', type=int, help='# points for test set', required=False, default=250)
parser.add_argument('--load', type=str, help='whether to load results', required=False, default='False')
parser.add_argument('--save_qnn', type=str, help='whether save trained qnns',
                    required=False, default='False')
parser.add_argument('--dataset', type=str, help='dataset to be used', required=False, default='fashion')
parser.add_argument('--optimizer', type=str, help='optimizer to be used', required=False, default='rms')
parser.add_argument('--model', type=str, help='model to train', required=False, default='pulsed')
parser.add_argument('--interface', type=str, help='interface', required=False, default='jax')
parser.add_argument('--allow_schedule', type=str, help='whether to allow schedules for optimizer',
                    required=False, default='False')
parser.add_argument('--realistic_gates', type=str, help='whether to use realistic gates for GateQNN',
                    required=False, default='True')

args = parser.parse_args()
realistic_gates = ast.literal_eval(args.realistic_gates)
n_trials = args.trials
epochs = int(args.epochs)
n_layers = int(args.n_layers)
n_qubits = int(args.n_qubits)
n_jobs = int(args.n_jobs)
n_train = int(args.n_train)
n_test = int(args.n_test)
point_dimension = int(args.point_dimension)
dataset = args.dataset
optimizer = args.optimizer
interface = args.interface
model = args.model
LOAD_RESULTS = ast.literal_eval(args.load)
allow_schedule = ast.literal_eval(args.allow_schedule)
save_qnn = ast.literal_eval(args.save_qnn)



EXP_RESULTS_PATH = os.path.join(get_root_path('Hardware-Adapted-Quantum-Machine-Learning'), f'data/results/tuning/{dataset}/{model}')
QNN_PATH = os.path.join(EXP_RESULTS_PATH, 'trained_qnn')
os.makedirs(EXP_RESULTS_PATH, exist_ok=True)
os.makedirs(QNN_PATH, exist_ok=True)


################ TUNING GRID #################################
LR_MIN, LR_MAX = 0.000001, 0.1
BETA1 = 0.9
BETA2 = 0.999
BATCH_SIZE = 24


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
        n_boundaries = trial.suggest_int("n_boundaries", 1, min(4, epochs - 1))  # at most 4 changes of lr
        lr_boundaries = sorted(trial.suggest_int(f"boundary_{i}", 1, epochs - 1) for i in range(n_boundaries))
        lr = [trial.suggest_float(f"lr_{i}", LR_MIN, LR_MAX, log=True) for i in range(n_boundaries + 1)]
        params = {"lr": lr, "lr_boundaries": lr_boundaries}
    return params


################ CREATE DATASET #################################
train_set, train_labels, test_set, test_labels = get_dataset(dataset,
                                                             n_train,
                                                             n_test,
                                                             interface,
                                                             points_dimension=point_dimension,
                                                             seed=SEED, scale=pi)

################ TUNING EXPERIMENT FUNCTIONS #################################
def train_and_evaluate(n_layers, n_epochs, batch_size, opt_params):
    print(f'Starting trial with: {n_layers = }, {n_epochs = }, {batch_size = }, {opt_params = }')

    # epochs_dict = {0: int(n_epochs / 3), -1: n_epochs}
    epochs_dict = n_epochs
    qnn = get_qnn(model, n_qubits, n_layers, realistic_gates, SEED, interface)

    df_train = qnn.train(data_points_train=train_set, data_labels_train=train_labels,
                         # data_points_test=test_set, data_labels_test=test_labels,     # Para que vaya mas rapido
                         n_epochs=epochs_dict,
                         batch_size=batch_size,
                         optimizer=optimizer,
                         optimizer_parameters=opt_params,
                         silent=True,  # To go faster
                         save_stats=False,  # To go faster
                         )

    train_loss = df_train.iloc[-1]['loss'].item(0)
    final_acc_test = qnn.get_accuracy(test_set, test_labels)
    final_acc_train = qnn.get_accuracy(train_set, train_labels)
    ic(final_acc_test, final_acc_train, train_loss)
    score = train_loss

    # Save qnn
    qnn_id = get_highest_id(QNN_PATH, 'qnn', f'{pickle_extension}') + 1
    if save_qnn:
        path = os.path.join(QNN_PATH, f'qnn_{qnn_id}.{pickle_extension}')
        qnn.save_qnn(path)

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
        save_dict_to_json(params_dict, path.replace(f'{pickle_extension}', 'json'))

    # Save stats to global df
    stat_row = pd.DataFrame([{
        'qnn_id': qnn_id,
        'train_loss': train_loss,
        'acc_train': final_acc_train,
        'acc_test': final_acc_test,
        'score': score
    }])
    global df_stats_partial
    df_stats_partial = pd.concat([df_stats_partial, stat_row], ignore_index=True)

    return score


def objective(trial):
    opt_params = suggest_optim_parameters(trial, try_boundaries=allow_schedule)  # TRY BOUNDARIES
    score = train_and_evaluate(n_layers, epochs, BATCH_SIZE, opt_params)

    return score


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


def get_exp_path(exp_id: int):
    path = os.path.join(EXP_RESULTS_PATH, f'results_{exp_id}.csv')
    return path


def save_results(exp_id: int, study, df_stats: pd.DataFrame, config_dict: dict):
    path = get_exp_path(exp_id)

    # Save results
    results = [{"trial_number": trial.number,
                "params": trial.params,
                "cost": trial.value}
               for trial in study.trials]
    df = pd.DataFrame(results).sort_values(by="cost", ascending=False)
    df.to_csv(path, index=False)

    # Save stats
    path_stats = os.path.join(os.path.dirname(path), f'stats_{exp_id}.csv')
    df_stats.sort_values(by="score", ascending=False).to_csv(path_stats, index=False)

    # Save experiment configurations
    path_config = os.path.join(EXP_RESULTS_PATH, f'config_{exp_id}.json')
    save_dict_to_json(config_dict, path_config)

def save_plot(qnn):
    path = get_exp_path(exp_id)
    path = path.replace('results_', 'plot_').replace('.csv', '.png')
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    visualize_predictions_2d(qnn, X=train_set, y=train_labels, ax=ax, title=f'`{qnn.model_name}', show=False)
    fig.tight_layout()
    fig.savefig(path)


def load_results(exp_id):
    path = get_exp_path(exp_id)
    df = pd.read_csv(path).sort_values(by="cost", ascending=False)
    result_cost = df['cost']
    result_dic = df['params']
    return result_dic.to_dict(), result_cost


################ MAIN #################################
if __name__ == '__main__':

    # load results
    if LOAD_RESULTS and os.path.exists(get_exp_path(exp_id=0)):  # load first results
        result_dic, result_cost = load_results(exp_id=0)
        print('Result loaded from disk')
    else:
        # run optimization
        print('Could not load from disk. Running tuning experiment.')

        study = optuna.create_study(direction="minimize", study_name='Tuning Experiment')
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        result_dic = study.best_params
        exp_id = get_highest_id(EXP_RESULTS_PATH) + 1
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
            "n_layers": n_layers,
            "n_qubits": n_qubits,
            "BETA1": BETA1,
            "BETA2": BETA2,
            "epochs": epochs,
            "BATCH_SIZE": BATCH_SIZE,
            'optimizer': optimizer,
            'save_qnn': save_qnn,
        }

        save_results(exp_id, study, df_stats_partial, exp_dict)



    print(f'Best Parameters: {result_dic},\n   Best Cost: {result_cost}')
    print('------')
    print('\n\n[END] Tuning experiment finished!')
