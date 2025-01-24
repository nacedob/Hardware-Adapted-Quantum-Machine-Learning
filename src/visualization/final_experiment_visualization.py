import os
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from src.visualization import gate_color, pulsed_color, mixed_color, medium_violet, dark_violet, light_violet
from src.visualization.utils import darken_color
from src.visualization import visualize_predictions_2d
from src.utils import increase_dimensions
from src.QNN import BaseQNN
from src.utils import get_root_path, load_json_to_dict
from src.experiments.config_exp import get_dataset
from typing import Sequence

root = get_root_path()


def plot_comparison_final_experiment(grouped_df,
                                     error_bars='fillvalues',
                                     error_metric='std',
                                     plot_type='plot',
                                     legend:str='individual',
                                     metrics: [str, list] = None,
                                     colors: Sequence[str] = None,
                                     fontsize=12,
                                     figsize=(20, 5),
                                     plot_best: bool = False,
                                     savefolder:str=None,
                                     savename:str=None,
                                     show: bool = True):
    """
    Plots the comparison of metrics for the configurations 'Gate', 'Pulsed', and 'Mixed'.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        error_bars (str or bool): 'fillvalues' to fill between mean ± error,
                                  'errorbar' for error bars, or False for no error visualization.
        error_metric (str): 'std' to use standard deviation, or 'minmax' to use min and max as error values.
        plot_type (str): Type of plot ('scatter', 'plot', 'bar').
        fontsize (int): Font size for all elements in the plots (legend, axis labels, title, ticks).
    """
    if error_bars is True:
        error_bars = 'fillvalues'

    if plot_best:
        if error_bars:
            error_bars = False

    # Fill NaN values in error metrics if error_bars is enabled
    if error_bars is not False:
        grouped_df = grouped_df.fillna(0)

    all_metrics = ["acc_train", "acc_test", "train_loss"]
    if metrics is None:
        metrics = all_metrics
    elif metrics == 'accuracies':
        metrics = ['acc_train', 'acc_test']
    else:
        if isinstance(metrics, str):
            metrics = [metrics]
        elif not isinstance(metrics, list):
            raise ValueError('metrics must be a list or a str')
        for m in metrics:
            assert m in all_metrics, f'Metric {m} not recognized. Available: {all_metrics}'

    # Create the figure
    datasets = grouped_df.index.get_level_values(0).unique()
    figs = {}
    for dataset in datasets:
        grouped_dataset = grouped_df.loc[dataset, :, :]  # dataset, any qubits, any layers
        range_qubits = grouped_dataset.index.get_level_values(0).unique()

        if len(range_qubits) == 2:
            fig = plt.figure(constrained_layout=True, figsize=figsize)
            (subfig1, subfig2) = fig.subfigures(2, 1)  # create 2x1 subfigures+
            ax1 = subfig1.subplots(1, len(metrics))  # create 1x2 subplots on subfig1
            ax2 = subfig2.subplots(1, len(metrics))  # create 1x2 subplots on subfig2
            subfig1.suptitle('Number of qubits: 1', fontsize=fontsize, color=darken_color(dark_violet, 0.8))
            subfig2.suptitle('Number of qubits: 2', fontsize=fontsize, color=darken_color(dark_violet, 0.8))
            ax = np.array([ax1, ax2]).flatten()
        else:
            fig, ax = plt.subplots(len(range_qubits), 3, figsize=figsize)
            ax = np.array(ax).flatten()

        for i_q, qubits in enumerate(range_qubits):
            grouped_df_filtered = grouped_dataset.loc[qubits, :]

            # Define all models
            configurations = ["gate", "pulsed", "mixed"]

            # Define colors and bar width
            if colors is None:
                colors = [gate_color, pulsed_color, mixed_color]
            markers = ['d', 's', 'o']
            bar_width = 0.25
            num_configs = len(configurations)

            # Create the plots
            for i, metric in enumerate(metrics):
                ax_ = ax[i + i_q * len(metrics)]

                # Plot for each configuration
                for j, config in enumerate(configurations):
                    if not plot_best:
                        y_plot = grouped_df_filtered[f"{config}_{metric}_mean"]
                    else:
                        y_plot = grouped_df_filtered[f"{config}_{metric}_max"]
                    if error_metric == 'std':
                        y_lower = y_plot - grouped_df_filtered[f"{config}_{metric}_std"]
                        y_upper = y_plot + grouped_df_filtered[f"{config}_{metric}_std"]
                    elif error_metric == 'minmax':
                        y_lower = grouped_df_filtered[f"{config}_{metric}_min"]
                        y_upper = grouped_df_filtered[f"{config}_{metric}_max"]
                    else:
                        raise ValueError("Invalid error_metric option. Use 'std' or 'minmax'.")

                    if plot_type == 'scatter':
                        ax_.scatter(grouped_df_filtered.index, y_plot, label=config.capitalize(), color=colors[j],
                                    marker=markers[j])
                    elif plot_type == 'plot':
                        if error_bars == 'errorbar':
                            ax_.errorbar(grouped_df_filtered.index, y_plot,
                                         yerr=[y_plot - y_lower, y_upper - y_plot],
                                         label=config.capitalize(), capsize=5, marker=markers[j], color=colors[j])
                        elif error_bars == 'fillvalues':
                            ax_.fill_between(grouped_df_filtered.index, y_lower, y_upper, alpha=0.15, color=colors[j])
                            ax_.plot(grouped_df_filtered.index, y_plot, label=config.capitalize(), marker=markers[j],
                                     color=colors[j])
                        else:
                            ax_.plot(grouped_df_filtered.index, y_plot, label=config.capitalize(), marker=markers[j],
                                     color=colors[j])
                    elif plot_type == 'bar':
                        x_positions = grouped_df_filtered.index + (j - (num_configs - 1) / 2) * bar_width
                        ax_.bar(x_positions, y_plot, width=bar_width, label=config.capitalize(), color=colors[j])

                # Set titles and labels
                title_map = {
                    'acc_train': 'Train Accuracy',
                    'acc_test': 'Test Accuracy',
                    'train_loss': 'Train Loss'
                }
                ax_.set_title(title_map.get(metric, metric.capitalize()), fontsize=fontsize, color=medium_violet)
                ax_.set_xlabel("Number of layers", fontsize=fontsize + 1, color=medium_violet)
                ax_.set_ylabel('Accuracy' if metric != 'train_loss' else 'Loss', fontsize=fontsize, color=medium_violet)
                ax_.set_xticks(grouped_df_filtered.index)
                ax_.tick_params(axis='x', colors=medium_violet)  # Change x-axis tick color to red
                ax_.tick_params(axis='y', colors=medium_violet)  # Change y-axis tick color to blue
                ax_.tick_params(axis='both', which='major', labelsize=fontsize, color=medium_violet)
                if legend == 'individual':
                    ax_.legend(fontsize=fontsize - 2)

                ax_.grid(linestyle=":", alpha=0.4, color='gray')

                if metric == 'train_loss':
                    ax_.set_yscale('log')
                else:
                    ax_.set_ylim([0.5, 1])
        # Row Title
        if '3d' in dataset:
            dataset_ = dataset[:-2]
        else:
            dataset_ = dataset
        if dataset_ in ['fashion', 'digits']:
            dataset_ = 'MNIST ' + dataset_.capitalize()
        elif dataset_ == 'shell':
            dataset_ = 'Spherical Shell'
        else:
            dataset_ = dataset_.capitalize()
        fig.suptitle(dataset_, fontsize=fontsize + 3)
        if legend != 'individual':
            ax_lines, ax_labels = ax_.get_legend_handles_labels()
            fig.legend(ax_lines, ax_labels, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=fontsize - 2)

        if savefolder:
            os.makedirs(savefolder, exist_ok=True)
            if savename is not None:
                path = f'{savefolder}/{savename}.png'
            else:
                path = f'{savefolder}/{dataset}.png'
            fig.savefig(path, bbox_inches='tight', dpi=700)

        if show:
            fig.show()

        figs[dataset] = [fig, ax]
    return figs


def plot_comparison_final_experiment_multi(grouped_df,
                                           error_bars='fillvalues',
                                           error_metric='std',
                                           plot_type='plot',
                                           title: str = None,
                                           colors: Sequence[str] = None,
                                           fontsize=12,
                                           figsize=(20, 5),
                                           plot_best: bool = False,
                                           separation_line: bool = False,
                                           show: bool = True):
    """
    Plots the comparison of metrics for the configurations 'Gate', 'Pulsed', and 'Mixed'.

    Args:
        grouped_df (pd.DataFrame): The dataframe containing the data.
        error_bars (str or bool): 'fillvalues' to fill between mean ± error,
                                  'errorbar' for error bars, or False for no error visualization.
        error_metric (str): 'std' to use standard deviation, or 'minmax' to use min and max as error values.
        plot_type (str): Type of plot ('scatter', 'plot', 'bar').
        fontsize (int): Font size for all elements in the plots (legend, axis labels, title, ticks).
    """
    if error_bars is True:
        error_bars = 'fillvalues'

    if plot_best:
        if error_bars:
            error_bars = False

    # Fill NaN values in error metrics if error_bars is enabled
    if error_bars is not False:
        grouped_df = grouped_df.fillna(0)

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    datasets = grouped_df.index.get_level_values(0).unique()
    # Count the number of subplots (#datasets * #qubits_dataset)
    n_qubits_per_set = grouped_df.groupby('dataset').apply(lambda group:
                                                           group.index.get_level_values('n_qubits').nunique())
    n_subfigures = n_qubits_per_set.sum() + len(datasets)  # Add extra rows for dataset titles
    subfigs = fig.subfigures(n_subfigures, 1)  # one for each number of qubits and dataset

    counter = 0
    for dataset in datasets:
        # Add a blank subplot row for the dataset title
        subfig_title = subfigs[counter]
        subfig_title.subplots_adjust(top=0.1, bottom=0.1)
        ax_title = subfig_title.subplots(1, 1)
        ax_title.axis('off')  # Hide axes for the title
        ax_title.text(0.5, 0.5, f"Dataset: {dataset.capitalize()}", ha='center', va='center',
                      fontsize=fontsize + 4, color=darken_color(dark_violet, 0.8),
                      transform=ax_title.transAxes)
        counter += 1

        # Process subplots for each dataset
        grouped_dataset = grouped_df.loc[dataset, :, :]  # dataset, any qubits, any layers
        range_qubits = grouped_dataset.index.get_level_values(0).unique()

        if len(range_qubits) == 2:
            subfig1, subfig2 = subfigs[counter], subfigs[counter + 1]
            ax1 = subfig1.subplots(1, 3)  # (train_acc, test_acc, train_loss)
            ax2 = subfig2.subplots(1, 3)  # (train_acc, test_acc, train_loss)
            subfig1.suptitle('Number of qubits: 1', fontsize=fontsize, color=darken_color(dark_violet, 0.8))
            subfig2.suptitle('Number of qubits: 2', fontsize=fontsize, color=darken_color(dark_violet, 0.8))
            ax = np.array([ax1, ax2]).flatten()
            subfig1.subplots_adjust(hspace=0.02)
            subfig1.subplots_adjust(hspace=0.02)

            # Add horizontal line to separate the two subfigures
            if separation_line:
                try:
                    ax_line = subfigs[counter + 1].subplots(1, 1)
                    ax_line.axhline(-0.2, color=light_violet, linewidth=3, linestyle=':')  # Horizontal line
                    ax_line.axis('off')
                except IndexError:
                    pass
            counter += 2
        else:
            fig, ax = plt.subplots(len(range_qubits), 3, figsize=figsize)
            subfig1 = subfigs[counter]
            ax1 = subfig1.subplots(1, 3)  # (train_acc, test_acc, train_loss)
            ax1 = np.array(ax).flatten()
            counter += 1

        for i_q, qubits in enumerate(range_qubits):
            grouped_df_filtered = grouped_dataset.loc[qubits, :]

            # Define the metrics to compare
            metrics = ["acc_train", "acc_test", "train_loss"]
            configurations = ["gate", "pulsed", "mixed"]

            # Define colors and bar width
            if colors is None:
                colors = [gate_color, pulsed_color, mixed_color]
            markers = ['d', 's', 'o']
            bar_width = 0.25
            num_configs = len(configurations)

            # Create the plots
            for i, metric in enumerate(metrics):
                ax_ = ax[i + i_q * 3]

                # Plot for each configuration
                for j, config in enumerate(configurations):
                    if not plot_best:
                        y_plot = grouped_df_filtered[f"{config}_{metric}_mean"]
                    else:
                        y_plot = grouped_df_filtered[f"{config}_{metric}_max"]
                    if error_metric == 'std':
                        y_lower = y_plot - grouped_df_filtered[f"{config}_{metric}_std"]
                        y_upper = y_plot + grouped_df_filtered[f"{config}_{metric}_std"]
                    elif error_metric == 'minmax':
                        y_lower = grouped_df_filtered[f"{config}_{metric}_min"]
                        y_upper = grouped_df_filtered[f"{config}_{metric}_max"]
                    else:
                        raise ValueError("Invalid error_metric option. Use 'std' or 'minmax'.")

                    if plot_type == 'scatter':
                        ax_.scatter(grouped_df_filtered.index, y_plot, label=config.capitalize(), color=colors[j],
                                    marker=markers[j], marker_size=3)
                    elif plot_type == 'plot':
                        if error_bars == 'errorbar':
                            ax_.errorbar(grouped_df_filtered.index, y_plot,
                                         yerr=[y_plot - y_lower, y_upper - y_plot],
                                         label=config.capitalize(), capsize=5, marker=markers[j], color=colors[j])
                        elif error_bars == 'fillvalues':
                            ax_.fill_between(grouped_df_filtered.index, y_lower, y_upper, alpha=0.15, color=colors[j])
                            ax_.plot(grouped_df_filtered.index, y_plot, label=config.capitalize(), marker=markers[j],
                                     color=colors[j])
                        else:
                            ax_.plot(grouped_df_filtered.index, y_plot, label=config.capitalize(), marker=markers[j],
                                     color=colors[j])
                    elif plot_type == 'bar':
                        x_positions = grouped_df_filtered.index + (j - (num_configs - 1) / 2) * bar_width
                        ax_.bar(x_positions, y_plot, width=bar_width, label=config.capitalize(), color=colors[j])

                # Set titles and labels
                title_map = {
                    'acc_train': 'Train Accuracy',
                    'acc_test': 'Test Accuracy',
                    'train_loss': 'Train Loss'
                }
                ax_.set_title(title_map.get(metric, metric.capitalize()), fontsize=fontsize, color=medium_violet)
                ax_.set_xlabel("Number of layers", fontsize=fontsize + 1, color=medium_violet)
                ax_.set_ylabel('Accuracy' if metric != 'train_loss' else 'Loss', fontsize=fontsize, color=medium_violet)
                ax_.set_xticks(grouped_df_filtered.index)
                ax_.tick_params(axis='x', colors=medium_violet)  # Change x-axis tick color to red
                ax_.tick_params(axis='y', colors=medium_violet)  # Change y-axis tick color to blue
                ax_.tick_params(axis='both', which='major', labelsize=fontsize, color=medium_violet)
                ax_.legend(fontsize=fontsize - 2)
                ax_.grid(linestyle=":", alpha=0.4, color='gray')

                if metric == 'train_loss':
                    ax_.set_yscale('log')
                else:
                    ax_.set_ylim([0.4, 1])
                    ax_.set_ylim([0.4, 1])
            # Row Title
            if title is not None:
                fig.suptitle(title, fontsize=fontsize + 3)

            if show:
                fig.show()

    return fig, ax


def visualize_layer_results(df,
                            folder: str,
                            model: str,
                            n_layer_visualize: int,
                            exp_id: int = 0,
                            process_dataset: bool = True,
                            ax=None,
                            seed: int = None,
                            show: bool = False):
    if model == 'gate':
        model_name = 'GateQNN'
    elif model == 'mixed':
        model_name = 'PulsedQNN_encoding_gate'
    elif model == 'pulsed':
        model_name = 'PulsedQNN_encoding_pulsed'
    else:
        raise ValueError("Invalid model option. Use 'gate', 'pulsed' or 'mixed'.")
    df_layer = df[df['n_layers'] == n_layer_visualize]
    if seed is None:
        seed = int(df_layer.sort_values(f'{model}_train_loss', ascending=True).iloc[0]['seed'])
    best_qnn_path = f'{folder}/trained_qnn/{model_name}/qnn_layers_{n_layer_visualize}_seed_{seed}.pkl'
    config_dict = load_json_to_dict(f'{folder}/config_{exp_id}.json')
    dataset = config_dict['dataset']
    p_d = config_dict['point_dimension']

    qnn = BaseQNN.load_pickle(best_qnn_path)

    x, y, _, _ = get_dataset(dataset, 500, 10, 'jax', points_dimension=p_d, seed=seed)

    if process_dataset:
        points_ = increase_dimensions(x, 3, interface='jax')
    else:
        points_ = x
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax = visualize_predictions_2d(qnn, X=points_, y=y, show=show, ax=ax)
    acc = qnn.get_accuracy(points_, y, process_points=not process_dataset)
    ax.set_title(f'{model.capitalize()} - Accuracy: {acc}')
    return ax



# TABLA DE LATEX

def grouped_to_latex(df, output_file=None, scale_factor:float= 0.8, title:str=None, label:str=None, colorful:bool=False):
    """
    Generates a compact and elegant LaTeX table comparing models with mean and standard deviation.

    Parameters:
       df (pd.DataFrame): Input DataFrame with multiindex (dataset, n_qubits, n_layers).
       output_file (str, optional): If specified, saves the table to a .tex file.
       caption (str): If specified, Caption for the LaTeX table.
       label (str): If specified, label for the LaTeX table.

    Returns:
       str: LaTeX table as a string.
    """


    if title is None:
        title = "Comparison of models with Mean and Standard Deviation for different datasets."
    if label is None:
        label = 'final_table'

    # Restablecemos el índice para trabajar más fácilmente
    df_ = df.reset_index()
    modelos = ['gate', 'mixed', 'pulsed']
    colors = ['gate_color', 'mixed_color', 'pulsed_color']
    colors_dict = dict(zip(modelos, colors))
    # Crear columnas necesarias para el formato \pm
    metricas = ['train_loss', 'acc_train', 'acc_test']
    for modelo in modelos:
        for metrica in metricas:
            mean_col = f"{modelo}_{metrica}_mean"
            std_col = f"{modelo}_{metrica}_std"
            new_col = f"{modelo}_{metrica}"
            if 'loss' in metrica:
                df_[new_col] = df_[mean_col].map("{:.3f}".format) + r"$\pm$" + df_[std_col].map("{:.3f}".format)
            else:
                df_[new_col] = df_[mean_col].map("{:.2f}".format) + r"$\pm$" + df_[std_col].map("{:.2f}".format)

    # Filtrar solo columnas relevantes
    columnas = ['dataset', 'n_qubits', 'n_layers'] + [f"{modelo}_{metrica}" for modelo in modelos for metrica in
                                                      metricas]
    df_ = df_[columnas]
    # Construcción de la tabla LaTeX con agrupaciones
    tabla_latex = ""
    grupos = df_.groupby(["dataset", "n_qubits"])

    grouped_dataset = df_.groupby(["dataset"])
    n_col_per_dataset = {}
    done_datasets = set()
    for dataset, grupo in grouped_dataset:           # aqui saca el numero de filas que tiene el dataset (para ambos qubits)
        n_col_per_dataset[dataset[0]] = len(grupos)
    ic(n_col_per_dataset)

    for (dataset, qubits), grupo in grupos:
        # Inicio de un grupo
        num_filas = len(grupo)
        if dataset in done_datasets:
            tabla_latex += " & "  # Espacio en blanco para el dataset y qubits en la segunda fila
        else:
            tabla_latex += f"\\multirow{{{n_col_per_dataset[dataset]}}}{{*}}{{\\textbf{{{dataset.replace('3d', '').capitalize()}}}}} & "

        tabla_latex += f"\\multirow{{{num_filas}}}{{*}}{{\\textbf{{{qubits}}}}} & "

        # Agregar las filas del grupo
        for i, fila in grupo.iterrows():
            if i > grupo.index[0]:  # Para las siguientes filas, no repetimos dataset y qubits
                tabla_latex += " & & "
            tabla_latex += f"{fila['n_layers']} & "
            if colorful:
                tabla_latex += " & ".join(
                    [f"\\textcolor{{{colors_dict[modelo]}}}{{{fila[f'{modelo}_{metrica}']}}}" for modelo in modelos for
                     metrica in metricas]
                ) + " \\\\\n"
            else:
                tabla_latex += " & ".join(
                    [fila[f"{modelo}_{metrica}"] for modelo in modelos for metrica in metricas]) + " \\\\\n"

        # Separador doble horizontal entre grupos
        if dataset in done_datasets:
            tabla_latex += "\\hline\\hline\n"
        else:
            tabla_latex += "\\cline{2-12}\n"
            done_datasets.add(dataset)

    # Cabecera multinivel
    col_sep_list = ['|c||']*(len(modelos)-1) + ['c|']
    if colorful:
        header_modelos = (
                f"\\multicolumn{{3}}{{|c|}}{{}} & " +  # Encabezado de modelos en la primera fila
                " & ".join([f"\\multicolumn{{3}}{{{col_sep}}}"
                            f"{{\\textcolor{{{colors_dict[modelo]}}}{{\\textbf{{{modelo.capitalize()}}}}}}}"
                            for col_sep, modelo in zip(col_sep_list, modelos)]) + " \\\\\n"
        )
    else:
        header_modelos = (
                f"\\multicolumn{{3}}{{|c|}}{{}} & " +  # Encabezado de modelos en la primera fila
                " & ".join([f"\\multicolumn{{3}}{{{col_sep}}}{{\\textbf{{{modelo.capitalize()}}}}}" for col_sep, modelo
                            in zip(col_sep_list, modelos)]) +
                " \\\\\n"
        )
    if colorful:
        header_metricas = (
                "\\textbf{Dataset} & \\textbf{Qubits} & \\textbf{Layers} & " +  # Dataset, Qubits y Layers en la segunda fila
                ''.join(
                    [f"\\textcolor{{{color}}}{{\\textbf{{Train Loss}}}} & "
                     f"\\textcolor{{{color}}}{{\\textbf{{Train Acc}}}} & "
                     f"\\textcolor{{{color}}}{{\\textbf{{Test Acc}}}} &" for color in colors])[:-1] +   # ese -1 quita el & sobrante
                " \\\\\n"
        )
    else:
        header_metricas = (
                "\\textbf{Dataset} & \\textbf{Qubits} & \\textbf{Layers} & " +  # Dataset, Qubits y Layers en la segunda fila
                " & ".join(["\\textbf{Train Loss} & \\textbf{Train Acc} & \\textbf{Test Acc}"] * len(modelos)) +
                " \\\\\n"
        )

    # Envolvemos la tabla en el entorno tabular de LaTeX
    tabla_latex = (
            "\\begin{table*}\n\\centering\n"
            f"\\scalebox{{{scale_factor}}}"+"{\\begin{tabular}{|l|c|c||" + "c|c|c||" * (len(modelos)-1) + "c|c|c|""}\n"
                                                                   "\\hline\n"
            + header_modelos + "\\hline\\hline\n"
            + header_metricas + "\\hline\\hline\n"
            + tabla_latex +
            "\\end{tabular}}\n"
            f"\\caption{{{title}.}}\n"
            f"\\label{{tab:{label}}}\n"
            "\\end{table*}"
    )

    # Guardamos en un archivo si se especifica
    if output_file:
        with open(output_file, "w") as f:
            f.write(tabla_latex)

    return tabla_latex