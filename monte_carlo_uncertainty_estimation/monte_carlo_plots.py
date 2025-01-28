import torch
import matplotlib.pyplot as plt
import torch


def plot_results(all_mean_probs, all_std_probs, all_labels):
    """
    Demonstrates how to plot:
      1) Histograms of predicted mean probabilities for each class.
      2) A scatter plot of mean probability vs. standard deviation for class 0.
    """
    # Convert tensors to NumPy arrays (if not already)
    mean_probs_np = all_mean_probs.cpu().numpy()
    std_probs_np = all_std_probs.cpu().numpy()
    labels_np = all_labels.cpu().numpy()

    # mean_probs_np has shape (8154, 2), std_probs_np has shape (8154, 2), labels_np has shape (8154,)

    # 1) Histograms of predicted mean probabilities
    plt.figure(figsize=(10, 4))

    # Histogram for Class 0 probabilities
    plt.subplot(1, 2, 1)
    plt.hist(mean_probs_np[:, 0], bins=30, alpha=0.7, color='blue')
    plt.title("Mean Probabilities for Class 0")
    plt.xlabel("Mean Probability")
    plt.ylabel("Count")

    # Histogram for Class 1 probabilities
    plt.subplot(1, 2, 2)
    plt.hist(mean_probs_np[:, 1], bins=30, alpha=0.7, color='orange')
    plt.title("Mean Probabilities for Class 1")
    plt.xlabel("Mean Probability")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig('histogram_class_probabilities.png')
    plt.show()

    # 2) Scatter plot: mean probability vs. standard deviation (for Class 0)
    plt.figure(figsize=(6, 5))
    plt.scatter(mean_probs_np[:, 0], std_probs_np[:, 0], alpha=0.5, c='green')
    plt.title("Mean vs. Standard Deviation (Class 0)")
    plt.xlabel("Mean Probability (Class 0)")
    plt.ylabel("Std Probability (Class 0)")
    plt.savefig('scatter_mean_vs_std_class_0.png')
    plt.show()

    # 3) Scatter plot: mean probability vs. standard deviation (for Class 1)
    plt.figure(figsize=(6, 5))
    plt.scatter(mean_probs_np[:, 1], std_probs_np[:, 1], alpha=0.5, c='red')
    plt.title("Mean vs. Standard Deviation (Class 1)")
    plt.xlabel("Mean Probability (Class 1)")
    plt.ylabel("Std Probability (Class 1)")
    plt.savefig('scatter_mean_vs_std_class_1.png')
    plt.show()

    # Optionally, compare predicted labels vs. true labels
    predictions = torch.argmax(all_mean_probs, dim=1).cpu().numpy()
    accuracy = (predictions == labels_np).mean()
    print(f"Overall accuracy: {accuracy:.2%}")


def load_mc_results(file_path):
    results = torch.load(file_path)
    all_mean_probs = results['mean_probs']
    all_std_probs = results['std_probs']
    all_labels = results['labels']
    return all_mean_probs, all_std_probs, all_labels


if __name__ == "__main__":
     all_means, all_std_probs, all_labels = load_mc_results('../mc_droput_results/mc_dropout_results.pth')

     # Show dimensions of the loaded data
     print(all_means.shape)
     print(all_std_probs.shape)
     print(all_labels.shape)

     # Plot the results
     plot_results(all_means, all_std_probs, all_labels)