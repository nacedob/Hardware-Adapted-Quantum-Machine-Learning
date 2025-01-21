from src.visualization import visualize_predictions_2d
from matplotlib import pyplot as plt

def save_model_predictions_plots(path:str,
                                 qnnPulsed,
                                 qnnGate,
                                 train_set,
                                 train_labels,
                                 dataset_name: str,
                                 pulsed_accuracy:float=None,
                                 gate_accuracy:float=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    title_pulsed = 'Pulsed'
    if pulsed_accuracy is not None:
        title_pulsed += f', Accuracy: {pulsed_accuracy:.2f}'
    title_gate = 'Gate'
    if gate_accuracy is not None:
        title_gate += f', Accuracy: {gate_accuracy:.2f}'
    visualize_predictions_2d(qnnPulsed, X=train_set, y=train_labels, title=title_pulsed, show=False, ax=ax[0], grid_points=50)
    visualize_predictions_2d(qnnGate, X=train_set, y=train_labels, title=title_gate, show=False, ax=ax[1], grid_points=50)
    ax[0].set_aspect("equal", adjustable="box")
    ax[1].set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.suptitle(f'Dataset: {dataset_name}')
    fig.savefig(path)