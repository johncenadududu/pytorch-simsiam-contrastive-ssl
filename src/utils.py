import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image


def setup_logging(log_filename="training.log"):
    """
    Configures logging to output to both console and a file.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Configure root logger
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(log_filepath),
                            logging.StreamHandler()
                        ])
    
    # Silence matplotlib's verbose logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.info(f"Logging setup complete. Log file: {log_filepath}")


def save_plot(history, title, filename):
    """
    Saves a plot of training and testing loss/accuracy.
    """
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=16)

        epochs = range(1, len(history['train_loss']) + 1)

        # Plot Train vs Test Loss
        ax1.plot(epochs, history['train_loss'], label='Train Loss')
        ax1.plot(epochs, history['test_loss'], label='Test Loss')
        ax1.set_title('Train vs Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot Train vs Test Accuracy
        ax2.plot(epochs, history['train_acc'], label='Train Accuracy')
        ax2.plot(epochs, history['test_acc'], label='Test Accuracy')
        ax2.set_title('Train vs Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.savefig(filepath)
        plt.close(fig)
        logging.info(f"Saved plot to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save plot {filename}: {e}")


def save_pretext_plot(history, title, filename):
    """
    Saves a plot for the pretext task training.
    """
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(title, fontsize=16)
        epochs_range = range(1, len(history['train_loss']) + 1)

        # Left Plot: Train vs Test Loss
        ax1.plot(epochs_range, history['train_loss'], label='Train Loss')
        ax1.plot(epochs_range, history['test_loss'], label='Test Loss')
        ax1.set_title('Train vs Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Right Plot: All Accuracy Metrics
        ax2.plot(epochs_range, history['train_acc_rot'], label='Train Rotation Acc')
        ax2.plot(epochs_range, history['test_acc_rot'], label='Test Rotation Acc', linestyle='--')
        ax2.plot(epochs_range, history['train_acc_shear'], label='Train Shear Acc')
        ax2.plot(epochs_range, history['test_acc_shear'], label='Test Shear Acc', linestyle='--')
        ax2.plot(epochs_range, history['train_acc_color'], label='Train Color Acc')
        ax2.plot(epochs_range, history['test_acc_color'], label='Test Color Acc', linestyle='--')
        ax2.set_title('Train vs Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend(loc='best')
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filepath)
        plt.close(fig)
        logging.info(f"Saved plot to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save plot {filename}: {e}")


def find_and_save_top_k_closest(model_backbone, train_loader, test_loader, device, filename, sample_size=5, k=10):
    """
    Finds and saves a plot of the top k closest samples from the train set
    for a few random samples from the test set, based on backbone features.
    """
    logging.info("Visualizing feature space: finding top-k closest samples...")
    model_backbone.eval()
    output_dir = "outputs"
    filepath = os.path.join(output_dir, filename)

    try:
        # 1. Extract features from the entire training dataset (our gallery)
        logging.debug("Extracting features from the training set (gallery)...")
        train_features = []
        train_images = []
        with torch.no_grad():
            for images, _ in tqdm(train_loader, desc="Gallery Features", leave=False):
                images = images.to(device)
                features = model_backbone(images).view(images.size(0), -1)
                train_features.append(features.cpu())
                
                # Un-normalize images for visualization
                for img in images:
                    img = img.cpu().numpy().transpose((1, 2, 0))
                    mean = np.array([0.5, 0.5, 0.5])
                    std = np.array([0.5, 0.5, 0.5])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    train_images.append(img)

        train_features = torch.cat(train_features, dim=0)

        # 2. Select random samples from the test set
        logging.debug(f"Selecting {sample_size} random test images (queries)...")
        test_subset_indices = random.sample(range(len(test_loader.dataset)), sample_size)
        test_subset = Subset(test_loader.dataset, test_subset_indices)
        query_loader = DataLoader(test_subset, batch_size=sample_size)
        query_images, _ = next(iter(query_loader))

        with torch.no_grad():
            query_features = model_backbone(query_images.to(device)).view(query_images.size(0), -1)

        # 3. Compute distances and find top k
        distances = torch.cdist(query_features.cpu(), train_features)
        top_k_indices = torch.topk(distances, k=k, dim=1, largest=False).indices

        # 4. Plot the results
        logging.debug("Plotting top-k results...")
        plt.figure(figsize=(20, 2 * sample_size))
        for i in range(sample_size):
            # Plot query image
            ax = plt.subplot(sample_size, k + 1, i * (k + 1) + 1)
            query_img_np = query_images[i].numpy().transpose((1, 2, 0))
            query_img_np = np.array([0.5, 0.5, 0.5]) * query_img_np + np.array([0.5, 0.5, 0.5])
            query_img_np = np.clip(query_img_np, 0, 1)
            ax.imshow(query_img_np)
            ax.set_title("Query")
            ax.axis('off')

            # Plot top k similar images
            for j in range(k):
                ax = plt.subplot(sample_size, k + 1, i * (k + 1) + 2 + j)
                retrieved_img = train_images[top_k_indices[i, j]]
                ax.imshow(retrieved_img)
                ax.set_title(f"Rank {j+1}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        logging.info(f"Saved top-k similarity plot to {filepath}")

    except Exception as e:
        logging.error(f"Failed to generate top-k plot: {e}")
