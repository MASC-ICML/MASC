
from matplotlib import pyplot as plt


def plot_metrics(metrics, title=""):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(title)
    
    # Losses
    axs[0, 0].plot(metrics["train_loss"], label='Train')
    axs[0, 0].plot(metrics["val_loss"], label='Validation')
    # axs[0, 0].plot(metrics["train_reg"], label='Train regularization loss')
    axs[0, 0].plot(metrics["test_loss"], label='Test')
    if 'baseline' in metrics:
        axs[0, 0].axhline(y=metrics['baseline']['test_loss'], color='r', linestyle='--', label='Baseline Test Loss')
    axs[0, 0].set_title("Losses")
    axs[0, 0].legend()
    # plt.show()
    
    # Regularizations
    axs[0, 1].plot(metrics["train_reg"][1:], label='Train regularization')
    axs[0, 1].set_title("Regularization")
    axs[0, 1].legend()
    # plt.show()

    # Acuuracies
    axs[0, 2].plot(metrics["train_acc"], label='Train')
    axs[0, 2].plot(metrics["val_acc"], label='Validation')
    axs[0, 2].plot(metrics["test_acc"], label='Test')
    if 'baseline' in metrics:
        axs[0, 2].axhline(y=metrics['baseline']['test_acc'], color='r', linestyle='--', label='Baseline')
        axs[0, 2].axhline(y=metrics['linesearch_baseline']['test_acc'], color='m', linestyle='--', label='Linesearch')
        
    axs[0, 2].set_title("Accuracy")
    axs[0, 2].legend()
    # plt.show()

    # Weights
    weights = [key for key in metrics.keys() if key.startswith("w_") and key[2:].isdigit()]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(weights)):
        axs[1, 0].plot(metrics["w_" + str(i)], label='w' + str(i), color=colors[i])
    axs[1, 0].plot(metrics["w_bias"], label='bias', color=colors[-1])
    if 'baseline' in metrics and 'w_bias' in metrics['baseline']:
        for i in range(len(weights)):
            axs[1, 0].axhline(y=metrics['baseline']["w_" + str(i)], color=colors[i], linestyle='--', label=f'Baseline w_{i}')
        axs[1, 0].axhline(y=metrics['baseline']["w_bias"], color=colors[-1], linestyle='--', label='Baseline Bias')
    axs[1, 0].set_title("Model weights")
    axs[1, 0].legend()
    # plt.show()

    # Prices
    axs[1, 1].plot(metrics["train_price"], label='Training Price')
    axs[1, 1].plot(metrics["test_price"], label='Test Price')
    if 'baseline' in metrics:
        axs[1, 1].axhline(y=metrics['baseline']["test_ps"], color='r', linestyle='--', label='Baseline Price')
    # y_max = np.quantile(np.array(metrics["train_price"]), 0.90) * 1.2
    # y_min = np.quantile(np.array(metrics["train_price"]), 0.10) * 0.8
    axs[1, 1].set_title("Price")
    axs[1, 1].legend()
    # plt.show()


    if 'val_tp' in metrics:
        # axs[1, 2].plot(metrics[f"test_tp"], label=f'Test Pos stayed', linestyle=':', color='g')
        axs[1, 2].plot(metrics[f"test_fn_moved"], label=f'Test Pos moved', linestyle='-', color='g')
        axs[1, 2].plot(metrics[f"test_fn_stayed"], label=f'Test Pos stuck', linestyle='--', color='g')
        axs[1, 2].plot(metrics[f"test_tn_moved"], label=f'Test Neg moved', linestyle='--', color='r')
        # axs[1, 2].plot(metrics[f"test_tn_moved"], label=f'Test Neg moved', linestyle='--', color='r')
        axs[1, 2].plot(metrics[f"test_tn_stayed"], label=f'Test Neg stuck', linestyle='-', color='r')
    #     # for mvmt_key in mvmts:
    #         # axs[1, 2].plot(metrics[f"train_{mvmt_key}"], label=f'Train {mvmt_key}', linestyle='-')
    #         # axs[1, 2].plot(metrics[f"val_{mvmt_key}"], label=f'Val {mvmt_key}', linestyle='-')
    #         # axs[1, 2].plot(metrics[f"test_{mvmt_key}"], label=f'Test {mvmt_key}', linestyle='-')
    #         # if 'baseline' in metrics:
    #         #     axs[1, 2].axhline(y=metrics['baseline'][mvmt_key], color='red', linestyle='--', label=f'Baseline {mvmt_key}')
    # else: # old keys
    #     axs[1, 2].plot(metrics["train_pos_moved"], label='Pos Moved Train', linestyle='-', color='blue')
    #     axs[1, 2].plot(metrics["val_pos_moved"], label='Pos Moved Val', linestyle='-', color='cyan')
    #     axs[1, 2].plot(metrics["train_neg_stayed"], label='Neg Stayed Train', linestyle='-', color='orange')
    #     axs[1, 2].plot(metrics["val_neg_stayed"], label='Neg Stayed Val', linestyle='-', color='yellow')
    #     if 'baseline' in metrics:
    #         axs[1, 2].axhline(y=metrics['baseline']["pos_moved_val"], color='red', linestyle='--', label='Baseline Pos Moved')
    #         axs[1, 2].axhline(y=metrics['baseline']["neg_stayed_val"], color='pink', linestyle='--', label='Baseline Neg Stayed')

    axs[1, 2].set_title("Buyers Movement")
    axs[1, 2].legend()
    axs[1, 2].set_title("Movements")
    axs[1, 2].legend()
    
    
    plt.show()
