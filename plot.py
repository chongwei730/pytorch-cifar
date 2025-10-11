import matplotlib.pyplot as plt
import pandas as pd
import os

save_dir = "./adam_base_lr_compare"

method1_name = "cosine"
method2_name = "line search armijo"

method1_data = pd.read_csv(os.path.join(save_dir, "cosine.csv"))
method2_data = pd.read_csv(os.path.join(save_dir, f"LineSearch_16384_Adam_armijo_1.0_log.csv"))

colors = {
    method1_name: "tab:blue",
    method2_name: "tab:orange"
}

fig, axs = plt.subplots(3, 1, figsize=(10, 15))


axs[0].plot(method1_data['epoch'], method1_data['lr'],
            color=colors[method1_name], linestyle='-', label=f'{method1_name} LR')
axs[0].step(method2_data['epoch'], method2_data['lr'],
            color=colors[method2_name], linestyle='-', where='post', label=f'{method2_name} LR')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Learning Rate')
axs[0].set_title('Learning Rate Comparison')
axs[0].legend()
axs[0].grid(True)


axs[1].plot(method1_data['epoch'], method1_data['train_loss'],
            color=colors[method1_name], linestyle='-', label=f'{method1_name} Train Loss')
axs[1].plot(method1_data['epoch'], method1_data['test_loss'],
            color=colors[method1_name], linestyle='--', label=f'{method1_name} Test Loss')
axs[1].plot(method2_data['epoch'], method2_data['train_loss'],
            color=colors[method2_name], linestyle='-', label=f'{method2_name} Train Loss')
axs[1].plot(method2_data['epoch'], method2_data['test_loss'],
            color=colors[method2_name], linestyle='--', label=f'{method2_name} Test Loss')

summary_text_loss = []
for name, data, color in [(method1_name, method1_data, colors[method1_name]),
                          (method2_name, method2_data, colors[method2_name])]:
    idx_train = data['train_loss'].idxmin()
    epoch_train = data['epoch'][idx_train]
    axs[1].scatter(epoch_train, data['train_loss'][idx_train], color=color, marker='o')
    summary_text_loss.append(f"{name} Train Min: {data['train_loss'][idx_train]:.3f} (Epoch {epoch_train})")

    idx_test = data['test_loss'].idxmin()
    epoch_test = data['epoch'][idx_test]
    axs[1].scatter(epoch_test, data['test_loss'][idx_test], color=color, marker='x')
    summary_text_loss.append(f"{name} Test Min: {data['test_loss'][idx_test]:.3f} (Epoch {epoch_test})")

axs[1].text(0.99, 0.5, "\n".join(summary_text_loss),
            transform=axs[1].transAxes, ha='right', va='center', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].set_title('Train Loss Comparison')
axs[1].legend()
axs[1].grid(True)


axs[2].plot(method1_data['epoch'], method1_data['train_acc'],
            color=colors[method1_name], linestyle='-', label=f'{method1_name} Train Acc')
axs[2].plot(method1_data['epoch'], method1_data['test_acc'],
            color=colors[method1_name], linestyle='--', label=f'{method1_name} Test Acc')
axs[2].plot(method2_data['epoch'], method2_data['train_acc'],
            color=colors[method2_name], linestyle='-', label=f'{method2_name} Train Acc')
axs[2].plot(method2_data['epoch'], method2_data['test_acc'],
            color=colors[method2_name], linestyle='--', label=f'{method2_name} Test Acc')

summary_text_acc = []
for name, data, color in [(method1_name, method1_data, colors[method1_name]),
                          (method2_name, method2_data, colors[method2_name])]:
    idx_train = data['train_acc'].idxmax()
    epoch_train = data['epoch'][idx_train]
    axs[2].scatter(epoch_train, data['train_acc'][idx_train], color=color, marker='o')
    summary_text_acc.append(f"{name} Train Max: {data['train_acc'][idx_train]:.3f} (Epoch {epoch_train})")

    idx_test = data['test_acc'].idxmax()
    epoch_test = data['epoch'][idx_test]
    axs[2].scatter(epoch_test, data['test_acc'][idx_test], color=color, marker='x')
    summary_text_acc.append(f"{name} Test Max: {data['test_acc'][idx_test]:.3f} (Epoch {epoch_test})")

axs[2].text(0.99, 0.5, "\n".join(summary_text_acc),
            transform=axs[2].transAxes, ha='right', va='center', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Accuracy')
axs[2].set_title('Train/Test Accuracy Comparison')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"1_comparison.png"))
plt.show()
