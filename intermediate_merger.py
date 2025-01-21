import os
from src.utils import get_root_path
import pandas as pd
from icecream import ic
import argparse
parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--dataset', type=str, required=False, default='fashion')
parser.add_argument('--id', type=int, required=False, default=0)

args = parser.parse_args()
root = get_root_path('Hardware-Adapted-Quantum-Machine-Learning')
dataset = args.dataset
exp_id = args.id
ic(dataset, exp_id)

folder = f'{root}/data/results/final_experiment/{dataset}'
intermediate_folder = f'{folder}/intermediate/exp_id_{exp_id}'

dfs = {}
for model in ['gate', 'mixed', 'pulsed']:
    dfs_model = pd.DataFrame([])
    model_path = f'{intermediate_folder}/{model}'
    for f in os.listdir(model_path):
        if f.endswith('.csv') and int(f[-5]) == exp_id:
            df = pd.read_csv(f'{model_path}/{f}').set_index(['n_qubits', 'n_layers', 'seed'])
            try:
                df = df.drop(columns=['qnn_id'], inplace=True)
            except KeyError:
                pass
            if df is None:
                continue
            df.columns = [f"{model}_{col}" for col in df.columns]
            dfs_model = pd.concat([dfs_model, df])
            pd.concat([dfs_model, df])
    dfs[model] = dfs_model

combined = pd.concat([df for df in dfs.values()], axis=1)

combined.to_csv(f'{folder}/results_{exp_id}_.csv')