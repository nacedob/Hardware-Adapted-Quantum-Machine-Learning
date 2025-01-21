"""
Este experimento solo hace el training de uno de los dos modelos para un dataset
Example:
    python -m src.experiments.pulsed_vs_gate --dataset=circles --model=gate
"""
from math import pi
from .config_exp import get_highest_id, get_dataset, get_qnn
from joblib import Parallel, delayed
import ast
import argparse
from src.utils import save_dict_to_json
import pandas as pd
from icecream import ic
from src.utils import get_root_path
import os
from src.QNN import PulsedQNN, GateQNN
from warnings import filterwarnings
from src.utils import pickle_extension
from src.visualization import visualize_predictions_2d

filterwarnings('ignore', category=FutureWarning)

SEED = 0

################ PROCESS ARGS FROM COMMAND LINE #################################
parser = argparse.ArgumentParser()

parser.add_argument('--n_train', type=int, help='# points for train set', required=False, default=400)
parser.add_argument('--n_test', type=int, help='# points for test set', required=False, default=200)
parser.add_argument('--n_qubits', type=int, help='# qubits', required=False, default=1)
parser.add_argument('--load', type=str, help='whether to load results', required=False, default='False')
parser.add_argument('--dataset', type=str, help='dataset to be used', required=False, default='fashion')
parser.add_argument('--interface', type=str, help='interface', required=False, default='jax')
parser.add_argument('--model', type=str, help='model to be trained', required=False, default='pulsed')
parser.add_argument('--n_layers', type=int, help='# layers to use for qnn', required=False, default=5)
parser.add_argument('--n_epochs', type=int, help='# epochs for training', required=False, default=30)
parser.add_argument('--lr', type=float, help='learning_rate', required=False, default=-1)
parser.add_argument('--n_trials', type=int, help='# repetitions of the training', required=False, default=4)
parser.add_argument('--point_dimension', type=int, help='Dimension of dataset', required=False, default=3)
parser.add_argument('--n_jobs', type=int, help='# workers', required=False, default=4)
parser.add_argument('--optimizer', type=str, help='optimizer to use', required=False, default='rms')
parser.add_argument('--realistic_gates', type=str, help='whether to use realistic gates for GateQNN',
                    required=False, default='True')
parser.add_argument('--save_qnn', type=str, help='whether to save qnn after training', required=False,
                    default='False')

args = parser.parse_args()
n_train = int(args.n_train)
n_test = int(args.n_test)
n_layers = int(args.n_layers)
n_epochs = int(args.n_epochs)
n_qubits = int(args.n_qubits)
n_trials = int(args.n_trials)
point_dimension = int(args.point_dimension)
n_jobs = int(args.n_jobs)
dataset = args.dataset
model = args.model
optimizer = args.optimizer
interface = args.interface
batch_size = 24
LOAD_RESULTS = ast.literal_eval(args.load)
realistic_gates = ast.literal_eval(args.realistic_gates)
save_qnn = ast.literal_eval(args.save_qnn)
lr = args.lr
if lr == -1:
    lr = 0.05

################ EXPERIMENT_FOLDERS #################################
EXP_RESULTS_PATH = os.path.join(get_root_path('Hardware-Adapted-Quantum-Machine-Learning'), f'data/results/train_qnn/{dataset}'+'/{}')
QNN_PATH = os.path.join(EXP_RESULTS_PATH, 'trained_qnn')
PARTIAL_PATH = os.path.join(EXP_RESULTS_PATH, 'partial')
os.makedirs(EXP_RESULTS_PATH.format('PulsedQNN_encoding_gate'), exist_ok=True)
os.makedirs(EXP_RESULTS_PATH.format('PulsedQNN_encoding_pulsed'), exist_ok=True)
os.makedirs(EXP_RESULTS_PATH.format('GateQNN'), exist_ok=True)
os.makedirs(PARTIAL_PATH.format('PulsedQNN_encoding_gate'), exist_ok=True)
os.makedirs(PARTIAL_PATH.format('PulsedQNN_encoding_pulsed'), exist_ok=True)
os.makedirs(PARTIAL_PATH.format('GateQNN'), exist_ok=True)

################ CREATE DATASET #################################
train_set, train_labels, test_set, test_labels = get_dataset(dataset,
                                                             n_train,
                                                             n_test,
                                                             interface,
                                                             points_dimension=point_dimension,
                                                             seed=SEED, scale=pi)


################ TUNING EXPERIMENT FUNCTIONS #################################
def train_and_evaluate(qnn, exp_desc: str):
    qnn_path = os.path.join(QNN_PATH.format(qnn.model_name),
                            f'qnn__{exp_desc}.{pickle_extension}')

    print(f'Training model {qnn.model_name}')
    df_train = qnn.train(data_points_train=train_set, data_labels_train=train_labels,
                         # data_points_test=test_set, data_labels_test=test_labels,     # Para que vaya mas rapido
                         n_epochs=n_epochs,
                         batch_size=batch_size,
                         optimizer=optimizer,
                         optimizer_parameters={'lr': lr},
                         silent=True,  # To go faster
                         save_stats=False,  # To go faster
                         )
    train_loss = df_train.iloc[-1]['loss'].item(0)
    final_acc_train = qnn.get_accuracy(train_set, train_labels)
    final_acc_test = qnn.get_accuracy(test_set, test_labels)
    ic(final_acc_train, final_acc_test, train_loss)
    if save_qnn:
        qnn.save_qnn(qnn_path)

        # Save qnn

        # save dict containing qnn parameters
        params_dict = {
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'final_accuracy_train': final_acc_train,
            'final_accuracy_test': final_acc_test,
            'train_loss': train_loss,
            'save_qnn': save_qnn,
        }
        save_dict_to_json(params_dict, qnn_path.replace(f'{pickle_extension}', 'json'))

    # Save stats to global df
    qnn_stats = pd.DataFrame([{
        'train_loss': train_loss,
        'acc_train': final_acc_train,
        'acc_test': final_acc_test,
        'qnn_name': qnn.model_name,
        'qnn_path': qnn_path,
    }])

    return qnn_stats


################ SAVE RESULTS #################################
def get_exp_path(exp_id: str):
    path = os.path.join(EXP_RESULTS_PATH, f'results_{exp_id}.csv')
    return path


def save_results_partial(exp_desc: str, df_stats: pd.DataFrame, config_dict: dict, qnn):
    path = os.path.join(PARTIAL_PATH.format(qnn.model_name), f'{exp_desc}.csv')
    # Save stats
    df_stats.to_csv(path, index=False)

    # Save experiment configurations
    path_config = os.path.join(PARTIAL_PATH.format(qnn.model_name), f'config_{exp_desc}.json')
    save_dict_to_json(config_dict, path_config)

    # Visualize predictions
    # path_fig = os.path.join(PARTIAL_PATH.format(qnn.model_name), f'predictions_{exp_desc}.png')
    # visualize_predictions_2d(qnn, train_set, train_labels, grid_points=200, title=f'{qnn.name}, dataset={dataset}',
    #                          save_path=path_fig,
    #                          show=False)


def load_results(exp_id):
    path = get_exp_path(exp_id).format(model)
    df = pd.read_csv(path).sort_values(by="acc_train", ascending=False)
    return df


def function_worker(seed: int):
    ic(seed)
    # Gate model
    qnn = get_qnn(model, n_qubits, n_layers, realistic_gates, seed, interface)

    exp_desc = (
        f'{dataset}__nQubits_{n_qubits}__nLayers_{n_layers}__nEpochs_{n_epochs}__batchSize_{batch_size}__lr_{lr}'
        f'_seed_{seed}')
    stats = train_and_evaluate(qnn, exp_desc)
    stats.index = [seed]
    stats.index.name = 'seed'
    # Save experiment results
    exp_dict = {
        "n_train": n_train,
        "n_test": n_test,
        "dataset": dataset,
        "lr": lr,
        "n_layers": n_layers,
        "n_qubits": n_qubits,
        "n_epochs": n_epochs,
        "seed": seed,
        "batch_size": batch_size,
        "point_dimension": point_dimension,
    }

    save_results_partial(exp_desc, stats, exp_dict, qnn)

    return stats


################ MAIN #################################
def main_experiment():
    # load results
    # if LOAD_RESULTS and os.path.exists(get_exp_path(exp_id=exp_desc).format(model)):  # load first results
    #     stats = load_results(exp_id=exp_desc)
    #     print('Results loaded from disk')
    # else:
    #     # run optimization
    #     print('Could not load from disk. Running training experiment.')

    results = Parallel(n_jobs=n_jobs)(delayed(function_worker)(seed) for seed in range(SEED, SEED + n_trials))
    results = pd.concat(results, axis=0)
    return results


if __name__ == "__main__":
    results = main_experiment()

    # Compute the mean of the results
    columns = ['acc_train', 'acc_test', 'train_loss']
    stats = pd.DataFrame({
        'Mean': results[columns].mean(),
        'Std': results[columns].std(),
        'Max': results[columns].max(),
        'Min': results[columns].min(),
    })
    ic(stats)

    #save stats
    if model.lower() == 'mixed':
        model_ = 'PulsedQNN_encoding_gate'
    elif model.lower() == 'pulsed':
        model_ = 'PulsedQNN_encoding_pulsed'
    elif model.lower() == 'gate':
        model_ = 'GateQNN'
    else:
        raise ValueError(f'Model name {model} does not match pulsed or gate')
    exp_path = EXP_RESULTS_PATH.format(model_)
    exp_id = get_highest_id(exp_path, pattern='results') + 1

    results_path = os.path.join(exp_path, f'results_{exp_id}.csv')
    stats_path = os.path.join(exp_path, f'stats_{exp_id}.csv')
    config_path = os.path.join(exp_path, f'config_{exp_id}.json')
    results.to_csv(results_path)
    stats.to_csv(stats_path, index=True)

    config_dict = {
        'model': model,
        'dataset': dataset,
        'n_qubits': n_qubits,
        'n_epochs': n_epochs,
        'n_layers': n_layers,
        'n_trials': n_trials,
        'lr': lr,
        'n_train': n_train,
        'n_test': n_test,
        'batch_size': batch_size,
        'point_dimension': point_dimension,
        'save_qnn': save_qnn,
    }
    save_dict_to_json(config_dict, config_path)
