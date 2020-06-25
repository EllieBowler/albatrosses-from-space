from matplotlib import pyplot as plt
import numpy as np


def plot_training_curve(total_train_loss, total_val_loss, save_path):
    plt.figure(figsize=(9, 6))
    plt.plot(total_train_loss, label='training loss')
    plt.plot(total_val_loss, label='validation loss')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()
    return


def plot_recall_precision(recall, precision, save_path, test_island):
    thresholds = np.linspace(0.05, 0.95, 19)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Recall Precision plots for {}'.format(test_island))
    axes[0].set_ylim([0, 1])
    axes[0].set_xlim([0, 1])
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[1].set_ylim([0, 1])
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Probability Threshold')
    axes[1].set_ylabel('Recall')
    axes[2].set_ylim([0, 1])
    axes[2].set_xlim([0, 1])
    axes[2].set_xlabel('Probability Threshold')
    axes[2].set_ylabel('Precision')

    axes[0].plot(recall, precision)
    axes[1].plot(thresholds, recall)
    axes[2].plot(thresholds, precision)

    fig.savefig(save_path, bbox_inches='tight')
    plt.close()
    return
