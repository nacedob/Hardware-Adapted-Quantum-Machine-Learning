"""
este script pasa por todas las carpetas de todos los datasets de final_Experiment y
los junta en un results.csv y results.json
"""

import os
from src.utils import get_root_path, load_json_to_dict, save_dict_to_json
import pandas as pd
from icecream import ic
import argparse
parser = argparse.ArgumentParser(description='Your script description')

default_files_id = list(range(20))

parser.add_argument('--dataset', type=str, required=False, default='all')
parser.add_argument('--ids', type=int, required=False, default=default_files_id)

args = parser.parse_args()
root = get_root_path('Hardware-Adapted-Quantum-Machine-Learning')
dataset = args.dataset
exp_ids = args.ids

if dataset == 'all':
    dataset_list = ['fashion', 'digits', 'sinus3d', 'helix', 'shell', 'corners3d']
else:
    dataset_list = [dataset]

all_df = pd.DataFrame([])

for dataset in dataset_list:
    folder = f'{root}/data/results/final_experiment/{dataset}'
    dfs = []
    configs = {}
    exp_ids_actual = []
    for f in os.listdir(folder):
        if f.endswith('.csv') and '[' not in f:
            if f[-5] == '_' or f[-5] == 's':    # la s es por si hay un results.csv
                continue
            exp_id = int(f[-5])
            if exp_id in exp_ids:
                dfs.append(pd.read_csv(f'{folder}/{f}').set_index(['n_qubits', 'n_layers', 'seed']))
                exp_ids_actual.append(exp_id)
        elif f.endswith('.json') and '[' not in f:
            if f[-6] == '_' or f[-6] == 'g':    # la g es por si hay un config.json
                continue
            exp_id = int(f[-6])
            if exp_id in exp_ids:
                config = load_json_to_dict(f'{folder}/{f}')
                configs[exp_id] = config
    # if len(exp_ids_actual) <= 1:
    #     continue
    combined = pd.concat(dfs, axis=0)
    combined.sort_values(['n_qubits', 'n_layers', 'seed'], inplace=True)
    combined.to_csv(f'{folder}/results.csv')
    combined['dataset'] = [dataset]*len(combined)
    all_df = pd.concat([all_df, combined])
    save_dict_to_json(configs, f'{folder}/config.json')
    ic(dataset, exp_ids_actual)

all_df.reset_index(inplace=True, drop=False)

all_df.set_index(['dataset', 'n_qubits', 'n_layers', 'seed'], inplace=True)
all_df.sort_index(inplace=True)
all_df.to_csv(f'{root}/data/final_plots/all.csv')
