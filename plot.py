import matplotlib.pyplot as plt
import pandas as pd


cosine_data = pd.read_csv('Cosine_log.csv')
line_search_data = pd.read_csv('LineSearch_4096_log.csv')

colors = {
    "Cosine": "tab:blue",
    "Line Search": "tab:orange"
}

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].plot(cosine_data['epoch'], cosine_data['lr'],
            color=colors["Cosine"], linestyle='-', label='Cosine LR')
axs[0].step(line_search_data['epoch'], line_search_data['lr'],
            color=colors["Line Search"], linestyle='-', where='post', label='Line Search LR')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Learning Rate')
axs[0].set_title('Learning Rate Comparison')
axs[0].legend()
axs[0].grid(True)


axs[1].plot(cosine_data['epoch'], cosine_data['train_loss'],
            color=colors["Cosine"], linestyle='-', label='Cosine Train Loss')
axs[1].plot(cosine_data['epoch'], cosine_data['test_loss'],
            color=colors["Cosine"], linestyle='--', label='Cosine Test Loss')
axs[1].plot(line_search_data['epoch'], line_search_data['train_loss'],
            color=colors["Line Search"], linestyle='-', label='Line Search Train Loss')
axs[1].plot(line_search_data['epoch'], line_search_data['test_loss'],
            color=colors["Line Search"], linestyle='--', label='Line Search Test Loss')


summary_text_loss = []
for name, data, color in [("Cosine", cosine_data, colors["Cosine"]),
                          ("Line Search", line_search_data, colors["Line Search"])]:
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
axs[1].set_title('Train/Test Loss Comparison')
axs[1].legend()
axs[1].grid(True)


axs[2].plot(cosine_data['epoch'], cosine_data['train_acc'],
            color=colors["Cosine"], linestyle='-', label='Cosine Train Acc')
axs[2].plot(cosine_data['epoch'], cosine_data['test_acc'],
            color=colors["Cosine"], linestyle='--', label='Cosine Test Acc')
axs[2].plot(line_search_data['epoch'], line_search_data['train_acc'],
            color=colors["Line Search"], linestyle='-', label='Line Search Train Acc')
axs[2].plot(line_search_data['epoch'], line_search_data['test_acc'],
            color=colors["Line Search"], linestyle='--', label='Line Search Test Acc')


summary_text_acc = []
for name, data, color in [("Cosine", cosine_data, colors["Cosine"]),
                          ("Line Search", line_search_data, colors["Line Search"])]:
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
plt.savefig("./comparison.png")
plt.show()
