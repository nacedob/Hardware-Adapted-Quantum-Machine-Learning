"""
Exte experimento tunea para un dataset y optimizer fijo para cada una de las posibles para los tres modelos
layers entre layers_min y layers_max y para 1 y 2 qubits, todas las posibilidades. El numero de épocas también es fijo
Es un experimento de fine tuning pero lo he llamado tuning_fine para que aparezca al lado de los otros en la carpeta.
El resultado de cada tuning es el lr.
"""
from time import time
from src.utils import (save_dict_to_json, print_in_green, print_in_blue, print_in_gray, load_pickle, save_pickle,
                         get_root_path)
import optuna
import pandas as pd
from icecream import ic
import argparse
import ast
from src.utils import pickle_extension
import re
import os
from warnings import filterwarnings
from .config_exp import get_dataset, get_qnn

filterwarnings('ignore', category=FutureWarning)

SEED = 0

################ TUNING PARAMS #################################
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, help='Number of trials to tune hyperparameters',
                    required=False, default=25)
parser.add_argument('--layers_min', type=int, help='Min number of layers to train', required=False, default=1)
parser.add_argument('--layers_max', type=int, help='Max number of layers to train', required=False, default=10)
parser.add_argument('--epochs', type=int, help='Number of epochs', required=False, default=30)
parser.add_argument('--n_jobs', type=int, help='Number of simultaneous trials', required=False, default=4)
parser.add_argument('--n_train', type=int, help='# points for train set', required=False, default=400)
parser.add_argument('--n_test', type=int, help='# points for test set', required=False, default=200)
parser.add_argument('--point_dimension', type=int, help='Dimension of dataset', required=False, default=3)
parser.add_argument('--load', type=str, help='whether to load results', required=False, default='False')
parser.add_argument('--log', type=str, help='logarithm sampling of lr', required=False, default='True')
parser.add_argument('--dataset', type=str, help='dataset to be used', required=False, default='fashion')
parser.add_argument('--optimizer', type=str, help='optimizer to be used', required=False, default='rms')
parser.add_argument('--realistic_gates', type=str, help='whether to use realistic gates for GateQNN',
                    required=False, default='True')
parser.add_argument('--save_qnn', type=str, help='whether to save trained qnn',
                    required=False, default='False')
parser.add_argument('--model', type=str, help='model to tune', required=False, default='all')

args = parser.parse_args()
n_trials = args.trials
epochs = int(args.epochs)
layers_min = int(args.layers_min)
layers_max = int(args.layers_max)
n_jobs = int(args.n_jobs)
n_train = int(args.n_train)
n_test = int(args.n_test)
point_dimension = int(args.point_dimension)
dataset = args.dataset
optimizer = args.optimizer
LOAD_RESULTS = ast.literal_eval(args.load)
realistic_gates = ast.literal_eval(args.realistic_gates)
save_qnn = ast.literal_eval(args.save_qnn)
log_sampling = ast.literal_eval(args.log)
model_ = args.model

if model_ == 'all':
    model_list = ['pulsed', 'gate', 'mixed']
else:
    model_list = [model_]
# model_list = ['gate']

root_path = get_root_path('Hardware-Adapted-Quantum-Machine-Learning')
EXP_RESULTS_PATH = os.path.join(root_path, f'data/results/tuning_fine/{optimizer}/{dataset}')
PARTIAL_RESULTS_PATH = os.path.join(EXP_RESULTS_PATH, f'{EXP_RESULTS_PATH}/{{}}_q_{{}}_l_{{}}')
QNN_PATH = os.path.join(PARTIAL_RESULTS_PATH, 'trained_qnn')
for m in model_list:
    for q in [1, 2]:
        for l in range(layers_min, layers_max + 1):
            os.makedirs(PARTIAL_RESULTS_PATH.format(m, q, l), exist_ok=True)
            os.makedirs(QNN_PATH.format(m, q, l), exist_ok=True)

################ TUNING GRID #################################
LR_MIN, LR_MAX = 0.00001, 0.10
BATCH_SIZE = 24

################ CREATE DATASET #################################
train_set, train_labels, test_set, test_labels = get_dataset(dataset,
                                                             n_train,
                                                             n_test,
                                                             'jax',
                                                             points_dimension=point_dimension,
                                                             seed=SEED)


################ TUNING EXPERIMENT FUNCTIONS #################################
def compute_score_tuning(model, n_qubits, n_layers, n_epochs, lr):
    """
    This is the function that computes the score for tuning for a fixed number of qubits, layers and model.
    """

    print(f'Starting trial with: {n_qubits = }, {model = }, {n_layers = }, {lr = }')
    qnn = get_qnn(model, n_qubits, n_layers, realistic_gates, SEED, 'jax')
    df_train = qnn.train(data_points_train=train_set, data_labels_train=train_labels,
                         # data_points_test=test_set, data_labels_test=test_labels,     # Para que vaya mas rapido
                         n_epochs=n_epochs,
                         batch_size=BATCH_SIZE,
                         optimizer=optimizer,
                         optimizer_parameters={'lr': lr},
                         silent=True,  # To go faster
                         save_stats=False,  # To go faster
                         )

    train_loss = df_train.iloc[-1]['loss'].item(0)
    final_acc_test = qnn.get_accuracy(test_set, test_labels)
    final_acc_train = qnn.get_accuracy(train_set, train_labels)
    ic(train_loss, final_acc_train, final_acc_test)
    score = train_loss

    # Save qnn
    qnn_folder = QNN_PATH.format(model, n_qubits, n_layers)
    qnn_id = get_highest_id(qnn_folder, 'qnn', f'{pickle_extension}') + 1
    path = os.path.join(qnn_folder, f'qnn_{qnn_id}.{pickle_extension}')
    if save_qnn:
        qnn.save_qnn(path)

        # save dict containing qnn parameters
        params_dict = {
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'lr': lr,
            'train_loss': train_loss,
            'final_accuracy_train': final_acc_train,
            'final_accuracy_test': final_acc_test,
            'dataset': dataset,
            'point_dimension': point_dimension,
            'n_epochs': n_epochs,
            'batch_size': BATCH_SIZE,
            'optimizer': optimizer,
            'save_qnn': save_qnn,
        }
        save_dict_to_json(params_dict, path.replace(f'{pickle_extension}', 'json'))

    # Save stats to global df
    stat_row = pd.DataFrame([{
        'qnn_id': qnn_id,
        'model': model,
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'lr': lr,
        'train_loss': train_loss,
        'acc_train': final_acc_train,
        'acc_test': final_acc_test,
        'score': score,
    }])
    global df_stats_partial
    df_stats_partial = pd.concat([df_stats_partial, stat_row], ignore_index=True)

    return score


def get_objective_tuning(model: str, n_qubits: int, n_layers: int):
    def objective(trial):
        lr = trial.suggest_float('lr', LR_MIN, LR_MAX, log=log_sampling)
        score = compute_score_tuning(model, n_qubits, n_layers, epochs, lr)
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


def get_exp_path(exp_id: int, model: str, n_qubits: int, n_layers: int):
    exp_path = PARTIAL_RESULTS_PATH.format(model, n_qubits, n_layers)
    path = os.path.join(exp_path, f'results_{exp_id}.csv')
    return path


def save_partial_results(exp_id: int, model, n_qubits, n_layers, study, df_stats: pd.DataFrame, config_dict: dict):
    path = get_exp_path(exp_id, model, n_qubits, n_layers)

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
    path_config = os.path.join(PARTIAL_RESULTS_PATH.format(model, n_qubits, n_layers), f'config_{exp_id}.json')
    save_dict_to_json(config_dict, path_config)


def load_partial_results(exp_id, model, n_qubits, n_layers):
    path = get_exp_path(exp_id, model, n_qubits, n_layers)
    df = pd.read_csv(path).sort_values(by="cost", ascending=False)
    best_params = df['params']
    best_cost = df['cost']
    return best_params.to_dict(), best_cost


################ MAIN #################################
if __name__ == '__main__':
    start_time = time()

    n_experiments = (layers_max - layers_min + 1) * 2 * n_trials
    expected_time = n_experiments * 2 / 60  # in hours
    print_in_blue(f'Number of experiments to be performed {n_experiments}. Expected time: {expected_time} hours')

    df_total = df_stats_partial = pd.DataFrame([],
                                               columns=['model', 'n_qubits', 'n_layers', 'best_params', 'best_score'])

    for model in model_list:
        print_in_blue(f'Starting experiment for model {model}')
        for n_qubits in [1, 2]:
            print_in_blue(f'Starting tuning for n_qubits={n_qubits}')
            for n_layers in range(layers_min, layers_max + 1):
                # load results
                if LOAD_RESULTS and os.path.exists(get_exp_path(0, model, n_qubits, n_layers)):  # load first results
                    best_params, best_score = load_partial_results(0, model, n_qubits, n_layers)
                    print_in_gray('Result loaded from disk')
                else:
                    # run optimization
                    print_in_gray('Could not load from disk. Running tuning experiment.')
                    df_stats_partial = pd.DataFrame([], columns=['qnn_id', 'model', 'n_qubits', 'n_layers', 'lr',
                                                                 'train_loss', 'acc_train', 'acc_test', 'score'])
                    print_in_blue(f'Starting tuning for n_layers={n_layers}')

                    objective_tuning = get_objective_tuning(model, n_qubits, n_layers)
                    study = optuna.create_study(direction="minimize", study_name='Tuning Experiment')
                    study.optimize(objective_tuning, n_trials=n_trials, n_jobs=n_jobs)

                    best_params = study.best_params
                    exp_id = get_highest_id(PARTIAL_RESULTS_PATH.format(model, n_qubits, n_layers)) + 1
                    best_score = study.best_value
                    exp_dict = {"model": model, "n_qubits": n_qubits, "n_jobs": n_jobs, "n_layers": n_layers,
                                'optimizer': optimizer, "LR_MIN": LR_MIN, "LR_MAX": LR_MAX, "n_trials": n_trials,
                                "dataset": dataset, "n_train": n_train, "n_test": n_test, "epochs": epochs,
                                "BATCH_SIZE": BATCH_SIZE, }

                    save_partial_results(exp_id, model, n_qubits, n_layers, study, df_stats_partial, exp_dict)

                print_in_green(f'Best Parameters: {best_params},\nBest Cost: {best_score}')
                print_in_green('[END] Tuning experiment finished!')

                # Save tuning results
                tuning_stats_df = pd.DataFrame([{'model': model, 'n_qubits': n_qubits, 'n_layers': n_layers,
                                                 'best_params': best_params, 'best_score': best_score}])

                df_total = pd.concat([df_total, tuning_stats_df], ignore_index=True)

    # Save all results
    results_id = get_highest_id(EXP_RESULTS_PATH, 'results', 'csv') + 1
    path = f'{EXP_RESULTS_PATH}/results_{results_id}.csv'
    ic(path, df_total)
    df_total.to_csv(path)


    # Save config experiment
    spent_time = time() - start_time
    config_dict = {'n_trials': n_trials, 'epochs': epochs, 'layers_min': layers_min, 'layers_max': layers_max,
                   'n_jobs': n_jobs, 'n_train': n_train, 'n_test': n_test, 'point_dimension': point_dimension,
                   'dataset': dataset, 'optimizer': optimizer, 'LOAD_RESULTS': LOAD_RESULTS,
                   'realistic_gates': realistic_gates, 'model_to_tune': model_, 'LR_MIN': LR_MIN,
                   'LR_MAX': LR_MAX,  'BATCH_SIZE': BATCH_SIZE, 'save_qnn': save_qnn,
                   'logarithm_sampling': log_sampling,
                   'total_experiment_time': spent_time}
    save_dict_to_json(config_dict, f'{EXP_RESULTS_PATH}/config_{results_id}.json')


    #### UNIFY RESULTS #####
    qubit_layer_df = {}

    FONTSIZE = 18

    for folder_qubit_layer in os.listdir(EXP_RESULTS_PATH):
        if folder_qubit_layer.endswith('.csv') or folder_qubit_layer.endswith('.json'):
            continue
        else:
            model, _, n_qubits, _, n_layers = folder_qubit_layer.split('_')
            try:
                df = pd.read_csv(f'{EXP_RESULTS_PATH}/{folder_qubit_layer}/stats_0.csv')
                if (n_qubits, n_layers) not in qubit_layer_df:
                    qubit_layer_df[(n_qubits, n_layers)] = [df]
                else:
                    qubit_layer_df[(n_qubits, n_layers)].append(df)

            except FileNotFoundError:
                continue

    for keys, values in qubit_layer_df.items():
        df = pd.concat(values)
        qubit_layer_df[keys] = df

    n = 5
    tolerance = 0.075  # si resultados de n=5 mejores filas difieren mas de esto, se desechan


    def filter_df(sub_df):
        threshold = sub_df['score'].max() * (1 - tolerance)
        return sub_df[sub_df['score'] >= threshold]


    tuning_save_folder = f'{root_path}/data/tuned_parameters/{optimizer}/{dataset}'
    os.makedirs(tuning_save_folder, exist_ok=True)

    dfs = []
    results = {}

    for (qubit, layers), df in qubit_layer_df.items():
        best_lines = df.drop(columns=['qnn_id', 'n_qubits', 'n_layers']).groupby('model').apply(
            lambda x: x.nlargest(n, 'score'), include_groups=False)

        # Filter results
        filtered_df = best_lines.droplevel(1, axis=0).groupby('model').apply(filter_df, include_groups=False).droplevel(
            0, axis=0).reset_index()
        filtered_df = filtered_df.groupby('model').agg(['mean', 'std'])
        dfs.append(filtered_df)

        # Save results
        for m in filtered_df.index:
            results[(m, qubit, layers)] = filtered_df.loc[m][('lr', 'mean')]


    # Save results
    save_pickle(results, f'{tuning_save_folder}/results_model_qubit_layer.{pickle_extension}')


    # Ahora sacar la media de todas
    df = pd.concat(dfs).reset_index()
    # Obtén las medias
    medias = df.xs('mean', level=1, axis=1)

    # Obtén la columna 'modelo'
    columna_modelo = df.xs('model', level=0, axis=1)
    columna_modelo.columns = ['model']

    # Concatena la columna 'modelo' y las medias
    df_all = pd.concat([columna_modelo, medias], axis=1).groupby('model').agg(['mean', 'std'])
    global_results = {}
    for m in df_all.index:
        global_results[m] = df_all.loc[m][('lr', 'mean')]
    save_pickle(global_results, f'{tuning_save_folder}/results_model_global.{pickle_extension}')


    # Unify results and save
    tuning_save_folder = f'{root_path}/data/tuned_parameters/{optimizer}'
    global_name = f'results_model_global.{pickle_extension}'
    qubit_layer_name = f'results_model_qubit_layer.{pickle_extension}'
    global_results = {}
    qubit_layer_results = {}
    path_ = os.path.join(tuning_save_folder, dataset)
    if os.path.exists(f'{path_}/{global_name}'):
        global_results[dataset] = load_pickle(f'{path_}/{global_name}')
    if os.path.exists(f'{path_}/{qubit_layer_name}'):
        qubit_layer_results[dataset] = load_pickle(f'{path_}/{qubit_layer_name}')
    # Extract mean of qubit_layer df and save it
    df = pd.DataFrame(qubit_layer_results)
    df.index.names = ('model', 'qubit', 'layer')
    df['media'] = df.mean(axis=1)
    df.drop(columns=[dataset], inplace=True)
    mean_qubit_layer = df.to_dict()['media']

    save_pickle(mean_qubit_layer, f'{tuning_save_folder}/{qubit_layer_name}')

    # Same for global lr
    df = pd.DataFrame(global_results)
    df.index.name = 'model'
    df['media'] = df.mean(axis=1)
    df.drop(columns=[dataset], inplace=True)
    mean_global = df.to_dict()['media']
    save_pickle(mean_global, f'{tuning_save_folder}/{global_name}')

    print_in_gray(f'[INFO] Experiment results saved in {path}')
    print_in_blue(f'Experimento terminado exitosamente. Los resultados para {dataset} se han guardado correctamente \n'
                  f'en "data/tuned_parameters/{dataset}". Pero NO se ha actualizado el dict global para tener en \n'
                  f'cuenta estos resultados. Para eso, corre el notebook "tuning_fine_unify"')
