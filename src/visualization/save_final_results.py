import os
from icecream import ic
import matplotlib.pyplot as plt
from src.utils import get_root_path, load_json_to_dict
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization import plot_comparison_final_experiment, visualize_layer_results, grouped_to_latex
import argparse
from ast import literal_eval
FONTSIZE = 28

root = get_root_path()

parser = argparse.ArgumentParser(description='Plot results for final experiment.')
parser.add_argument('--dataset', type=str, default='all', help='Dataset for which to plot results.')
parser.add_argument('--id', type=str, default=None, help='Exp id to analyze.')
args = parser.parse_args()
dataset = args.dataset
exp_id = args.id

datasets = None
if dataset == 'all':
    datasets = ['fashion', 'digits', 'sinus3d', 'helix', 'shell', 'corners3d']
    exp_ids = [''] * len(datasets)  # aqui meter a mano los que interesen

if datasets is None:
    if '[' not in dataset and '(' not in dataset:
        datasets = [dataset]
        if exp_id is None:
            exp_ids = [0]
        else:
            exp_ids = [exp_id]
    else:
        datasets = literal_eval(dataset)
        exp_ids = literal_eval(exp_id)

grouped_df = pd.DataFrame([])
df_all = pd.DataFrame([])
for set, id in zip(datasets, exp_ids):
    if id != '':
        id = f'_{id}'
    # Cargar el CSV
    folder = f'{root}/data/results/final_experiment/{set}/'
    try:
        config = load_json_to_dict(f"{folder}/config{id}.json")
    except FileNotFoundError:
        config = {'n_qubits': [1, 2]}
    df_results = pd.read_csv(f"{folder}/results{id}.csv")
    # Add dataset id to grouped_df
    df_results['dataset'] = set
    grouped_df_set = df_results.drop(columns=["seed"], inplace=False).groupby(['dataset', 'n_qubits', "n_layers"]).agg(["mean", "std", "min", "max"])

    grouped_df_set.columns = ['_'.join(col).strip() for col in grouped_df_set.columns]
    # ic(config)
    # if isinstance(list(config.values())[0], dict):
    #     n_qubits = []
    #     for dic in config.values():
    #         if isinstance(dic, dict):
    #             n_qubits.append(dic['n_qubits'][0])
    #     config["n_qubits"] = n_qubits

    df_results.set_index('dataset', inplace=True)
    df_results.index.name = 'dataset'
    df_all = pd.concat([df_all, df_results])
    
    # Count
    
    grouped_df = pd.concat([grouped_df, grouped_df_set])

df_all = df_all.sort_index(ascending=True)
df_all.to_csv(f'{root}/data/final_plots/all{id}.csv')
grouped_df.to_csv(f'{root}/data/final_plots/grouped{id}.csv')
grouped_to_latex(grouped_df, output_file=f'{root}/data/final_plots/table.tex', title='Results for different datasets',
                 scale_factor=0.9, colorful=True)

# Example usage

figs = plot_comparison_final_experiment(grouped_df,
                                        error_bars=True,
                                        error_metric='std',
                                        plot_type='plot',
                                        plot_best=False,
                                        fontsize=FONTSIZE,
                                        metrics='accuracies',
                                        legend='common',
                                        figsize=(17, 11),
                                        savefolder=f'{root}/data/final_plots/',
                                        show=False)
