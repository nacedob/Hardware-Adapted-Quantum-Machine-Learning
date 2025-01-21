import pandas as pd
from config_exp import RESULTS_FOLDER
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn
import ast
from config_exp import get_PLOT_RESULT
from icecream import ic

def get_data_experiment(experiment_id: int):
    return pd.read_csv(f'{RESULTS_FOLDER}/results_{experiment_id}.csv', index_col=0)


def filter_df_by_hardware(results_df):
    results_df['g'] = results_df['g'].apply(str)
    results_df['omega'] = results_df['omega'].apply(str)
    grouped = results_df.groupby(['g', 'omega'])
    filtered_dfs = {group: group_df for group, group_df in grouped}
    return filtered_dfs



def plot_results_filtered(experiment_id:int=None, results_df:pd.DataFrame=None):
    if experiment_id is None and results_df is None:
        raise ValueError('Either experiment_id or results_df should be provided.')
    elif experiment_id is not None and results_df is None:
        results_df = get_data_experiment(experiment_id)
    elif results_df is not None and experiment_id is not None:
        pass  # En este caso de usa directamente el df y se espera que este bien asignado con su id
        # warn('Provided both experiment_id and results_df. Using results_df directly.', stacklevel=2, category=UserWarning)

    PLOT_RESULTS = get_PLOT_RESULT()

    ids = []
    dfs_dict = {}
    hardwares = []

    def change_keys(key: str):
        ω = (np.array(ast.literal_eval(key[1])) / (2 * np.pi)).round(2).tolist()
        g = (np.array(ast.literal_eval(key[0])) / (2 * np.pi)).round(2).tolist()
        return f'{ω=}\n{g=}'


    filtered_dict_df = {change_keys(key): value for key, value in filter_df_by_hardware(results_df).items()}

    plt.figure()
    fig, ax = plt.subplots(1)
    fontsize = 8

    axis_margin = 0.5
    aux = []
    color = 'black'
    for i, (hardware, df) in enumerate(filtered_dict_df.items()):
        cost_vec = df['cost'].to_numpy()
        aux.append(cost_vec)
        ids = df.index.values
        ax.scatter(np.ones_like(cost_vec) * i, cost_vec, color=color)
        # add label
        for j in range(len(cost_vec)):
            ax.text(x=i+0.1, y=cost_vec[j], s=str(ids[j]), color=color, ha='left', va='center', fontsize=fontsize)

    ax.set_title('Results by hardware')

    len_filtered = len(filtered_dict_df)
    y_data = np.array(aux).flatten()

    ylim = min(y_data.min()*0.8, 1e-4)
    plt.fill_between([-axis_margin, len(filtered_dict_df) - axis_margin], [1e-3] * 2, ylim, alpha=0.2, color='lightskyblue')
    plt.xticks(range(len_filtered), list(filtered_dict_df.keys()), rotation=0, fontsize=fontsize)
    plt.xlim(-axis_margin, len_filtered - axis_margin)
    plt.ylim(ylim, 1 + 2 *
             axis_margin)
    plt.ylabel('Cost')
    plt.title('Best configurations results')
    plt.yscale('log')
    plt.tight_layout()

    # Save results
    fig_name = f'{RESULTS_FOLDER}/'
    if experiment_id is not None:
        fig_name += f'results_{experiment_id}_'
    fig_name += 'configuration_results.png'
    plt.savefig(fig_name)

    if PLOT_RESULTS:
        plt.show()

def plot_best_results(experiment_id:int=None, results_df:pd.DataFrame=None):
    if experiment_id is None and results_df is None:
        raise ValueError('Either experiment_id or results_df should be provided.')
    elif experiment_id is not None and results_df is None:
        results_df = get_data_experiment(experiment_id)
    elif results_df is not None and experiment_id is not None:
        pass # En este caso de usa directamente el df y se espera que este bien asignado con su id
        # warn('Provided both experiment_id and results_df. Using results_df directly.', stacklevel=2, category=UserWarning)

    PLOT_RESULTS = get_PLOT_RESULT()

    ids = []
    min_dfs = []
    hardwares = []
    for hardware, df in filter_df_by_hardware(results_df).items():
        ω = (np.array(ast.literal_eval(hardware[1]))/(2*np.pi)).round(2).tolist()
        g = (np.array(ast.literal_eval(hardware[0]))/(2*np.pi)).round(2).tolist()
        hardwares.append(f'{ω=}\n{g=}')


        min_cost_df = df[df['cost'] == df['cost'].min()]
        if len(min_cost_df) >= 1:
            min_cost_df = min_cost_df.iloc[[np.random.randint(0, len(min_cost_df))]]   # take any of them
        min_id = min_cost_df.index.values[0]
        min_dfs.append(min_cost_df)
        ids.append(min_id)

    min_dfs = pd.concat(min_dfs)
    l = len(min_dfs)
    round_func = lambda x: np.round(x,3)
    z0s = list(map(round_func, min_dfs['z0s'].to_numpy()))
    z1s = list(map(round_func, min_dfs['z1s'].to_numpy()))

    # PLOT RESULTS
    x_data = range(l)
    y_data = min_dfs['cost'].values

    # plot options
    fontsize = 8
    axis_margin = 0.5
    ylim = min(y_data.min()*0.8, 1e-4)

    plt.figure()
    plt.scatter(x_data, y_data)
    for i, (id_, cost_) in enumerate(zip(ids, y_data)):
        plt.annotate(f'id={id_}', (x_data[i], y_data[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=fontsize)
        z_str = f'z0={z0s[i]}\nz1={z1s[i]}'
        plt.annotate(z_str, (x_data[i], y_data[i]), textcoords="offset points", xytext=(15, 0), ha='left', fontsize=fontsize, color='gray')

    # axis and plot configurations
    plt.xticks(x_data, hardwares, rotation=0, fontsize=fontsize)
    plt.fill_between([-axis_margin, l-axis_margin], [1e-3] * 2, ylim, alpha=0.2, color='lightskyblue')
    plt.xlim(-axis_margin, l-axis_margin)
    plt.ylim(ylim, 1+2*
             axis_margin)
    plt.ylabel('Cost')
    plt.title('configurations results')
    plt.yscale('log')
    plt.tight_layout()

    if PLOT_RESULTS:
        plt.show()

    fig_name = f'{RESULTS_FOLDER}/'
    if experiment_id is not None:
        fig_name += f'results_{experiment_id}_'
    fig_name += 'best_configuration.png'
    plt.savefig(fig_name)


def plot_results(experiment_id:int = None, results_df:pd.DataFrame=None):
    plot_best_results(experiment_id, results_df)
    plot_results_filtered(experiment_id, results_df)


if __name__ == '__main__':
    plot_results_filtered(0)

