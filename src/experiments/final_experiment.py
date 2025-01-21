"""
This is the final experiment of the master's program. I take a dataset and perform tuning for a fixed number of layers 
(n_layers). Then, I sample randomly from the same dataset and train. I do this for n_seeds iterations. 
Additionally, the performance is studied with respect to the number of layers.
The models to be used are the gate-based model, the mixed model, and the 100% pulses model.
"""
from math import pi
from src.utils import save_dict_to_json, print_in_blue, print_in_gray, print_in_green, get_root_path
import optuna
import pandas as pd
from icecream import ic
from time import time
import argparse
from joblib import Parallel, delayed
import ast
from src.utils import pickle_extension
from src.QNN.BaseQNN import BaseQNN
from .config_exp import get_highest_id, get_dataset, get_qnn, get_optimal_lr, get_optimal_opt_parameters, get_eqk
import os
from warnings import filterwarnings, warn

filterwarnings('ignore', category=FutureWarning)

################ TUNING PARAMS #################################
parser = argparse.ArgumentParser()
parser.add_argument('--layers_min', type=int, help='#layers min to be compared', required=False, default=1)
parser.add_argument('--layers_max', type=int, help='#layers max to be compared', required=False, default=10)
parser.add_argument('--n_qubits', type=str, help='#qubits to use for both models', required=False, default='all')
parser.add_argument('--tuning', help='whether to tune model before each experiment', required=False, default='False')
parser.add_argument('--trials_tuning', type=float, required=False, default=50,
                    help='Number of trials to tune hyperparameters for each model')
parser.add_argument('--n_jobs', type=int, help='Number of simultaneous tunning', required=False, default=4)
parser.add_argument('--n_seeds', type=int, help='Number of seeds to perform experiment', required=False, default=5)
parser.add_argument('--starting_seed', type=int, help='starting seed', required=False, default=0)
parser.add_argument('--n_train', type=int, help='# points for train set', required=False, default=500)
parser.add_argument('--n_test', type=int, help='# points for test set', required=False, default=250)
parser.add_argument('--n_epochs', type=int, help='# epochs', required=False, default=30)
parser.add_argument('--point_dimension', type=int, help='Dimension of dataset', required=False, default=3)
parser.add_argument('--load', type=str, help='whether to load results', required=False, default='False')
parser.add_argument('--dataset', type=str, help='dataset to be used', required=False, default='fashion')
parser.add_argument('--optimizer', type=str, help='optimizer to be used', required=False, default='rms')
parser.add_argument('--interface', type=str, help='interface', required=False, default='jax')
parser.add_argument('--save_qnn', type=str, help='whether save trained qnns',
                    required=False, default='False')
parser.add_argument('--lr', type=float, help='learning_rate', required=False, default=-1)
parser.add_argument('--realistic_gates', type=str, help='whether to use realistic gates for GateQNN',
                    required=False, default='False')
parser.add_argument('--eqk', type=str, help='step further and train eqk',
                    required=False, default='False')
parser.add_argument('--use_stored_tuning', type=str, help='whether to stored tuning results for lr',
                    required=False, default='False')

args = parser.parse_args()
realistic_gates = ast.literal_eval(args.realistic_gates)
n_qubits = args.n_qubits
layers_min = args.layers_min
layers_max = args.layers_max
n_trials = args.trials_tuning
n_jobs = int(args.n_jobs)
n_train = int(args.n_train)
n_test = int(args.n_test)
n_epochs = int(args.n_epochs)
point_dimension = int(args.point_dimension)
n_seeds = int(args.n_seeds)
starting_seed = int(args.starting_seed)
dataset = args.dataset
optimizer = args.optimizer
interface = args.interface
LOAD_RESULTS = ast.literal_eval(args.load)
save_qnn = ast.literal_eval(args.save_qnn)
tuning_bool = ast.literal_eval(args.tuning)
eqk_bool = ast.literal_eval(args.eqk)
use_stored_tuning = ast.literal_eval(args.use_stored_tuning)
lr = args.lr

if n_qubits == 'all':
    n_qubits = [1, 2]
elif n_qubits == '1':
    n_qubits = [1]
elif n_qubits == '2':
    n_qubits = [2]

if eqk_bool:
    if not save_qnn:
        warn('If eqk is True, save_qnn must be True. It has been changed', category=UserWarning, stacklevel=2)
    save_qnn = True

empty_df = pd.DataFrame([], columns=['n_qubits', 'n_layers', 'seed', 'train_loss', 'acc_train', 'acc_test', 'lr'])


################ EXPERIMENT PATHS #################################
EXP_RESULTS_PATH = os.path.join(get_root_path('Hardware-Adapted-Quantum-Machine-Learning'),
                                f'data/results/final_experiment/{dataset}')
QNN_PATH = os.path.join(EXP_RESULTS_PATH, 'trained_qnn/')
os.makedirs(EXP_RESULTS_PATH, exist_ok=True)
os.makedirs(f'{EXP_RESULTS_PATH}/tuning', exist_ok=True)
INTERMEDIATE_FOLDER = os.path.join(EXP_RESULTS_PATH, f'intermediate')
os.makedirs(INTERMEDIATE_FOLDER, exist_ok=True)

################ TUNING GRID #################################
LR_MIN, LR_MAX = 0.00005, 0.05
BETA1 = 0.9
BETA2 = 0.999
BATCH_SIZE = 24

if not tuning_bool and use_stored_tuning:
    tuned_qubits_layer, tuned_global = get_optimal_lr(optimizer, dataset)
else:
    tuned_qubits_layer, tuned_global = None, None


################ CREATE DATASET #################################

################ TUNING EXPERIMENT FUNCTIONS #################################
def get_tuning_score(model: str, seed, qubits: int, n_layers, n_epochs, batch_size, opt_params):
    print(f'Starting trial with: {dataset = }, {qubits = }, {n_layers = }, {seed = }, {n_epochs = }, {opt_params = }')

    qnn = get_qnn(model, qubits, n_layers, realistic_gates, seed, interface)
    df_train = qnn.train(data_points_train=train_set, data_labels_train=train_labels,
                         # data_points_test=test_set, data_labels_test=test_labels,     # To go faster
                         n_epochs=n_epochs,
                         batch_size=batch_size,
                         optimizer=optimizer,
                         optimizer_parameters=opt_params,
                         silent=True,  # To go faster
                         save_stats=False,  # To go faster
                         )

    train_loss = df_train.iloc[-1]['loss'].item(0)
    final_acc_test = qnn.get_accuracy(test_set, test_labels)
    final_acc_train = qnn.get_accuracy(train_set, train_labels)

    # Save qnn



    stat_row = pd.DataFrame([{
        'n_qubits': qubits,
        'n_layers': n_layers,
        'seed': seed,
        'train_loss': train_loss,
        'acc_train': final_acc_train,
        'acc_test': final_acc_test,
        'lr': opt_params['lr'],
    }])
    global tuning_dfs
    global tuned_qnn
    if tuning_bool:
        tuning_dfs[model] = pd.concat([tuning_dfs[model], stat_row], ignore_index=True)
        current_cost = tuned_qnn[model][1]
        if train_loss < current_cost:
            tuned_qnn[model] = (qnn, current_cost)
    return train_loss


def objective_tuning(trial, seed, qubits: int, n_layers: int, model: str):
    lr_ = trial.suggest_float('lr', LR_MIN, LR_MAX, log=True)
    opt_params = {'lr': lr_}
    score = get_tuning_score(model, seed, qubits, n_layers, n_epochs, BATCH_SIZE, opt_params)
    return score


def tune_model(model: str, seed, qubits: int, n_layers: int, exp_id: int):
    print(f'[TUNING] Starting tuning experiment for model: {model.upper()}')
    study = optuna.create_study(direction="minimize", study_name=model)
    objective_gate = lambda trial: objective_tuning(trial, seed, qubits, n_layers, model)
    study.optimize(objective_gate, n_trials=n_trials, n_jobs=n_jobs)
    exp_dict = {
        "model": model,
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "n_train": n_train,
        "n_test": n_test,
        "dataset": dataset,
        "point_dimension": point_dimension,
        "n_layers": n_layers,
        "LR_MIN": LR_MIN,
        "LR_MAX": LR_MAX,
        "n_qubits": qubits,
        "BETA1": BETA1,
        "BETA2": BETA2,
        "n_epochs": n_epochs,
        "BATCH_SIZE": BATCH_SIZE,
        "realistic_gates": realistic_gates,
    }

    save_results_tuning(exp_id, qubits, n_layers, study, exp_dict, model)
    params_tuned = study.best_params
    print(f'[TUNING] Finished tuning experiment for model: {model.upper()}')
    return params_tuned


################ TRAINING FUNCTIONS #################################
def train_and_evaluate(qnn, exp_id: str, n_qubits: int, n_layers: int, batch_size: int, lr: float):
    print(f'[TRAINING] Training model {qnn.model_name}')


    df_train = qnn.train(data_points_train=train_set, data_labels_train=train_labels,
                         # data_points_test=test_set, data_labels_test=test_labels,     # Para que vaya mas rapido
                         n_epochs=n_epochs,
                         batch_size=batch_size,
                         optimizer_parameters={'lr': lr},
                         silent=True,  # To go faster
                         save_stats=False,  # To go faster
                         )

    train_loss = df_train.iloc[-1]['loss'].item(0)
    final_acc_test = qnn.get_accuracy(test_set, test_labels)
    final_acc_train = qnn.get_accuracy(train_set, train_labels)
    ic(final_acc_test, final_acc_train, train_loss)
    if save_qnn:
        if qnn.model_name == 'GateQNN':
            model = 'gate'
        elif qnn.model_name == 'PulsedQNN_encoding_gate':
            model = 'mixed'
        elif qnn.model_name == 'PulsedQNN_encoding_pulsed':
            model = 'pulsed'
        else:
            raise ValueError(f'{qnn.model_name} not recognized')
        qnn_path = get_qnn_path(model, n_qubits, n_layers, seed)
        qnn.save_qnn(qnn_path)

        # save dict containing qnn parameters
        params_dict = {
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': optimizer,
            'final_accuracy_train': final_acc_train,
            'final_accuracy_test': final_acc_test,
            'train_loss': train_loss,
            'seed': seed,
            'model': qnn.model_name,
            'dataset': dataset,
            "realistic_gates": realistic_gates,
        }
        save_dict_to_json(params_dict, qnn_path.replace(f'{pickle_extension}', 'json'))
    else:
        qnn_path = None

    # Save stats to global df

    if qnn.model_name == 'GateQNN':
        model = 'gate'
    elif qnn.model_name == 'PulsedQNN_encoding_gate':
        model = 'mixed'
    elif qnn.model_name == 'PulsedQNN_encoding_pulsed':
        model = 'pulsed'
    else:
        raise ValueError(f'{qnn.model_name} not recognized')

    qnn_stats = pd.DataFrame([{
        'model': model,
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'seed': seed,
        'train_loss': train_loss,
        'acc_train': final_acc_train,
        'acc_test': final_acc_test,
        'lr': lr,
        'epochs': str(n_epochs),
        'dataset': dataset,
        'optimizer': optimizer,
        'qnn_path': qnn_path,
    }])
    qnn_stats.set_index('n_layers', inplace=True)

    return qnn_stats


def get_qnn_stats(model, opt_params):
    qnn = get_qnn(model, qubits, n_layer, realistic_gates, seed, interface)
    df_stats = train_and_evaluate(qnn, exp_id, qubits, n_layer, BATCH_SIZE,
                                  opt_params['lr'])
    return df_stats


################ SAVE RESULTS #################################
def get_exp_path(exp_id: int):
    path = os.path.join(EXP_RESULTS_PATH, f'results_{exp_id}.csv')
    return path

def get_qnn_path(model, n_qubits, n_layers, seed):
    qnn_folder = f'{QNN_PATH.format(exp_id)}/{model}/exp_{exp_id}'
    qnn_path = os.path.join(qnn_folder, f'qnn_q_{n_qubits}_l_{n_layers}_s_{seed}.{pickle_extension}')
    return qnn_path

def save_results_tuning(exp_id: int, qubits, n_layer: int, study, config_dict: dict, model: str):
    path = get_exp_path(exp_id)
    path = path.replace('results_', f'tuning/{exp_id}/{model}_q_{qubits}_l_{n_layer}_s_{seed}')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save results
    results = [{"trial_number": trial.number,
                "params": trial.params,
                "cost": trial.value,
                'optimizer': optimizer}
               for trial in study.trials]
    df = pd.DataFrame(results).sort_values(by="cost", ascending=True)
    df.to_csv(path, index=False)

    # Save experiment configurations
    path_config = os.path.join(EXP_RESULTS_PATH, f'tuning//{exp_id}/config_tuning.json')
    save_dict_to_json(config_dict, path_config)


def save_results(df_stats: pd.DataFrame, exp_id: int):
    path = get_exp_path(exp_id)
    df_stats.to_csv(path)


def load_results(path):
    df = pd.read_csv(path)
    # ic(df)
    # df.set_index('qnn_name', inplace=True)
    return df


################ MAIN #################################
if __name__ == '__main__':
    start_experiment = time()
    exp_id = get_highest_id(EXP_RESULTS_PATH) + 1
    ic(exp_id)
    os.makedirs(os.path.join(QNN_PATH.format(exp_id), 'gate'), exist_ok=True)
    os.makedirs(os.path.join(QNN_PATH.format(exp_id), 'mixed'), exist_ok=True)
    os.makedirs(os.path.join(QNN_PATH.format(exp_id), 'pulsed'), exist_ok=True)
    # run optimization
    gate_path = os.path.join(f'{INTERMEDIATE_FOLDER}/exp_id_{exp_id}', 'gate')
    mixed_path = os.path.join(f'{INTERMEDIATE_FOLDER}/exp_id_{exp_id}', 'mixed')
    pulsed_path = os.path.join(f'{INTERMEDIATE_FOLDER}/exp_id_{exp_id}', 'pulsed')
    eqk_path = os.path.join(f'{INTERMEDIATE_FOLDER}/exp_id_{exp_id}', 'eqk')
    os.makedirs(gate_path, exist_ok=True)
    os.makedirs(mixed_path, exist_ok=True)
    os.makedirs(pulsed_path, exist_ok=True)
    df_stats_partial = pd.DataFrame()

    # Tuning experiment
    for qubits in n_qubits:
        for n_layer in range(layers_min, layers_max + 1):
            df_gate_intermediate = pd.DataFrame()
            df_mixed_intermediate = pd.DataFrame()
            df_pulsed_intermediate = pd.DataFrame()
            if eqk_bool:
                df_eqk_intermediate = pd.DataFrame()

            for seed in range(starting_seed, starting_seed + n_seeds):

                if tuning_bool:
                    tuning_dfs = {
                        'pulsed': empty_df.copy(),
                        'mixed': empty_df.copy(),
                        'gate': empty_df.copy(),
                    }

                    tuned_qnn = {
                        'pulsed': [None, 10000],   # stored_qnn, best_cost
                        'mixed': [None, 10000],   # stored_qnn, best_cost
                        'gate': [None, 10000],   # stored_qnn, best_cost
                    }

                gate_path_partial = os.path.join(gate_path,
                                                 f'qubits={qubits}layers={n_layer}_seed_{seed}_exp_id_{exp_id}.csv')
                mixed_path_partial = os.path.join(mixed_path,
                                                  f'qubits={qubits}layers={n_layer}_seed_{seed}_exp_id_{exp_id}.csv')
                pulsed_path_partial = os.path.join(pulsed_path,
                                                   f'qubits={qubits}layers={n_layer}_seed_{seed}_exp_id_{exp_id}.csv')

                # Get Dataset
                train_set, train_labels, test_set, test_labels = get_dataset(dataset,
                                                                             n_train,
                                                                             n_test,
                                                                             interface,
                                                                             points_dimension=point_dimension,
                                                                             seed=seed, scale=pi)

                if (LOAD_RESULTS and os.path.exists(gate_path_partial)
                        and os.path.exists(mixed_path_partial) and os.path.exists(pulsed_path_partial)):
                    print('---------')
                    print(
                        f'[INFO] QNN Loaded from path - Experiment for QUBITS: {qubits} - LAYERS: {n_layer} - SEED {seed}')
                    print('---------')
                    df_stats_gate = load_results(gate_path_partial)
                    df_stats_gate.set_index('n_layers', inplace=True)
                    df_stats_mixed = load_results(mixed_path_partial)
                    df_stats_mixed.set_index('n_layers', inplace=True)
                    df_stats_pulsed = load_results(pulsed_path_partial)
                    df_stats_pulsed.set_index('n_layers', inplace=True)

                else:
                    print('---------')
                    print_in_blue(f'[INFO] Running experiment for QUBITS: {qubits} - LAYERS: {n_layer} - SEED {seed}')
                    print('---------')
                    if tuning_bool:
                        params_pulsed = tune_model('pulsed', seed, qubits, n_layer, exp_id)
                        params_mixed = tune_model('mixed', seed, qubits, n_layer, exp_id)
                        params_gate= tune_model('gate', seed, qubits, n_layer, exp_id)
                        # get best tuning results for each model
                        df_stats_gate = tuning_dfs['gate'].sort_values('train_loss', ascending=True).iloc[[0]]
                        df_stats_pulsed = tuning_dfs['pulsed'].sort_values('train_loss', ascending=True).iloc[[0]]
                        df_stats_mixed = tuning_dfs['mixed'].sort_values('train_loss', ascending=True).iloc[[0]]

                        try:
                            df_stats_gate.set_index('n_layers', inplace=True, drop=True)
                        except:
                            pass
                        try:
                            df_stats_mixed.set_index('n_layers', inplace=True, drop=True)
                        except:
                            pass
                        try:
                            df_stats_pulsed.set_index('n_layers', inplace=True, drop=True)
                        except:
                            pass

                        # Save qnns
                        if save_qnn:
                            for model, (qnn, best_cost) in tuned_qnn.items():
                                qnn_path = get_qnn_path(model, qubits, n_layer, seed)
                                qnn.save_qnn(qnn_path)

                                params_dict = {
                                    'n_qubits': n_qubits,
                                    'n_layers': n_layer,
                                    'lr': lr,
                                    'train_loss': best_cost,
                                    'dataset': dataset,
                                    'point_dimension': point_dimension,
                                    'n_epochs': n_epochs,
                                    'batch_size': BATCH_SIZE,
                                    'optimizer': optimizer,
                                    'save_qnn': save_qnn,
                                    "realistic_gates": realistic_gates,
                                }
                                save_dict_to_json(params_dict, qnn_path.replace(f'{pickle_extension}', 'json'))


                        print('[TUNING] Tunning finished')
                    else:
                        if use_stored_tuning:
                            params_pulsed = \
                                get_optimal_opt_parameters(tuned_qubits_layer, tuned_global, 'pulsed', qubits, n_layer)
                            params_mixed = \
                                get_optimal_opt_parameters(tuned_qubits_layer, tuned_global, 'mixed', qubits, n_layer)
                            params_gate = \
                                get_optimal_opt_parameters(tuned_qubits_layer, tuned_global, 'gate', qubits, n_layer)
                        else:
                            params_pulsed = {'lr': 0.00045}
                            params_mixed = {'lr': 0.00045}
                            params_gate = {'lr': 0.05}

                        print('[TRAINING] Starting training and testing')

                        models = ['gate', 'mixed', 'pulsed']
                        model_opt_params = [params_gate, params_mixed, params_pulsed]
                        results = Parallel(n_jobs=n_jobs)(
                            delayed(get_qnn_stats)(model, opt_params) for model, opt_params in zip(models, model_opt_params)
                        )

                        for result_df in results:
                            if 'GateQNN' == result_df['qnn_name'].unique()[0]:
                                df_stats_gate = result_df
                            elif 'PulsedQNN_encoding_gate' == result_df['qnn_name'].unique()[0]:
                                df_stats_mixed = result_df
                            elif 'PulsedQNN_encoding_pulsed' == result_df['qnn_name'].unique()[0]:
                                df_stats_pulsed = result_df
                            else:
                                ic(result_df, result_df['qnn_name'].unique()[0])
                                raise ValueError('Unrecognized model')

                    # Save stats
                    df_stats_gate.to_csv(gate_path_partial)
                    df_stats_mixed.to_csv(mixed_path_partial)
                    df_stats_pulsed.to_csv(pulsed_path_partial)

                # Save in the df_partial
                df_stats_gate['n_layers'] = df_stats_gate.index
                df_stats_mixed['n_layers'] = df_stats_mixed.index
                df_stats_pulsed['n_layers'] = df_stats_pulsed.index
                df_stats_gate.set_index(['n_qubits', 'n_layers', 'seed'], inplace=True, drop=True)
                df_stats_mixed.set_index(['n_qubits', 'n_layers', 'seed'], inplace=True, drop=True)
                df_stats_pulsed.set_index(['n_qubits', 'n_layers', 'seed'], inplace=True, drop=True)
                df_gate_intermediate = pd.concat([df_gate_intermediate, df_stats_gate], axis=0)
                df_mixed_intermediate = pd.concat([df_mixed_intermediate, df_stats_mixed], axis=0)
                df_pulsed_intermediate = pd.concat([df_pulsed_intermediate, df_stats_pulsed], axis=0)


                if eqk_bool:
                    eqk_partial_path = f'{eqk_path}/qubit={qubits}layer={n_layer}_seed_{seed}_exp_id_{exp_id}.csv'
                    if (LOAD_RESULTS and os.path.exists(eqk_partial_path)):
                        print(f'[INFO] EQK results loaded from {eqk_partial_path}')
                        eqk_stats = pd.read_csv(eqk_partial_path)
                    else:
                        eqk_stats = {}
                        for mdl in ['gate', 'mixed', 'pulsed']:
                            qnn_path = get_qnn_path(mdl, qubits, n_layer, seed)
                            qnn = BaseQNN.load_qnn(qnn_path)
                            eqk = get_eqk(qnn)
                            df_eqk = eqk.train(train_set, train_labels, test_set, test_labels, silent=False)
                            acc_train_eqk = df_eqk['train_accuracy']
                            acc_test_eqk = df_eqk['test_accuracy']
                            eqk_stats[f'acc_train_{mdl}'] = acc_train_eqk
                            eqk_stats[f'acc_test_{mdl}'] = acc_test_eqk
                        multi_index = pd.MultiIndex.from_tuples([(qubits, n_layer, seed),],
                                                                names=['n_qubits', 'n_layers', 'seed'])
                        eqk_stats = pd.DataFrame(eqk_stats, index=multi_index)
                        ic(eqk_stats)
                        eqk_stats.to_csv(eqk_partial_path)
                    df_eqk_intermediate = pd.concat([df_eqk_intermediate, eqk_stats])

            # Merge stats
            compare_columns = {'train_loss', 'acc_train', 'acc_test', 'lr', 'seed'}
            cols_to_drop = set(df_gate_intermediate.columns).difference(compare_columns)
            try:
                df_gate_intermediate.drop(columns=cols_to_drop, inplace=True)
                df_mixed_intermediate.drop(columns=cols_to_drop, inplace=True)
                df_pulsed_intermediate.drop(columns=cols_to_drop, inplace=True)
            except:
                pass
            df_gate_intermediate.columns = [f"gate_{col}" for col in df_gate_intermediate.columns]
            df_mixed_intermediate.columns = [f"mixed_{col}" for col in df_mixed_intermediate.columns]
            df_pulsed_intermediate.columns = [f"pulsed_{col}" for col in df_pulsed_intermediate.columns]
            if eqk_bool:
                df_eqk_intermediate.columns = [f"eqk_{col}" for col in df_eqk_intermediate.columns]
                df_partial = pd.concat(
                    [df_gate_intermediate, df_pulsed_intermediate, df_mixed_intermediate, df_eqk_intermediate],
                    axis=1)
            else:
                df_partial = pd.concat(
                    [df_gate_intermediate, df_pulsed_intermediate, df_mixed_intermediate],
                    axis=1)

            df_stats_partial = pd.concat([df_stats_partial, df_partial], axis=0)

            # save intermediate results
            layer_path_df = os.path.join(INTERMEDIATE_FOLDER, f"exp_id_{exp_id}/qubit_{qubits}_layer_{n_layer}.csv")
            df_partial.to_csv(layer_path_df)

            # config layer to json
            config_layer_path_json = os.path.join(INTERMEDIATE_FOLDER,
                                                  f"exp_id_{exp_id}/qubit_{qubits}_layer_{n_layer}.json")
            config_layer = {
                'n_qubits': qubits,
                'layers_min': layers_min,
                'layers_max': layers_max,
                'n_trials': n_trials,
                'n_jobs': n_jobs,
                'n_train': n_train,
                'n_test': n_test,
                'point_dimension': point_dimension,
                'n_seeds': n_seeds,
                'dataset': dataset,
                'optimizer': optimizer,
                'interface': interface,
                'LOAD_RESULTS': LOAD_RESULTS,
                'tuning_bool': tuning_bool,
                'lr': lr,
                'n_epochs': n_epochs,
                "realistic_gates": realistic_gates,
            }
            save_dict_to_json(config_layer, config_layer_path_json)

    # Save experiment configuration and results
    path = get_exp_path(exp_id)
    df_stats_partial.to_csv(path)
    ic(df_stats_partial)
    spent_time = time() - start_experiment
    time_per_experiment = spent_time / len(df_stats_partial)
    config_layer = {
        'n_qubits': n_qubits, 'layers_min': layers_min, 'layers_max': layers_max, 'n_trials': n_trials,
        'n_jobs': n_jobs, 'n_train': n_train, 'n_test': n_test, 'point_dimension': point_dimension,
        'n_seeds': n_seeds, 'dataset': dataset, 'optimizer': optimizer, 'interface': interface, 'n_epochs': n_epochs,
        'LOAD_RESULTS': LOAD_RESULTS, 'tuning_bool': tuning_bool, 'lr': lr,
        'spent_time': spent_time, 'time_per_experiment': time_per_experiment,
    }
    save_dict_to_json(config_layer, path.replace('results_', 'config_').replace('.csv', '.json'))

    print_in_blue('\n\n[END] Experiment finished!')
    print_in_blue(f'[INFO] Total time spent:         {spent_time:.2f} seconds')
    print_in_blue(f'[INFO] Mean time per experiment: {time_per_experiment:.2f} seconds')
    print_in_blue(f'[INFO] Experiment results saved to {path}')
