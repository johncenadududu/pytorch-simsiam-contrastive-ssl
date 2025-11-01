import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import logging
import os
import json
from tqdm import tqdm

# Adjust path to import from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import SimSiam, negative_cosine_similarity_stopgradient, ClassificationModel
from src.data_loader import get_simsiam_loader, get_baseline_loaders
from src.utils import setup_logging, save_plot

def train_simsiam(model, train_loader, optimizer, scheduler, device, epochs):
    """
    Training loop for SimSiam pre-training.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [SimSiam Pretrain]", unit="batch")
        
        for (x1, x2), _ in progress_bar:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()

            # Forward pass for both views
            z1, p1 = model(x1)
            z2, p2 = model(x2)

            # Calculate loss (symmetric)
            loss = negative_cosine_similarity_stopgradient(p1, z2) / 2 + \
                   negative_cosine_similarity_stopgradient(p2, z1) / 2
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}, "
                     f"Current LR: {scheduler.get_last_lr()[0]:.6f}")

def train_linear_probe(classifier_model, train_loader, test_loader, optimizer, criterion, device, epochs, train_size, test_size):
    """
    Training loop for the linear probe (frozen backbone).
    """
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': []
    }

    for epoch in range(epochs):
        classifier_model.train()
        train_loss, correct_train, total_train = 0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Linear Probe]", unit="batch")
        
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            total_train += labels.size(0)
            correct_train += (outputs.argmax(1) == labels).sum().item()

        # Testing
        classifier_model.eval()
        test_loss, correct_test, total_test = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = classifier_model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                total_test += labels.size(0)
                correct_test += (outputs.argmax(1) == labels).sum().item()

        epoch_train_loss = train_loss / train_size
        epoch_train_acc = 100 * correct_train / total_train
        epoch_test_loss = test_loss / test_size
        epoch_test_acc = 100 * correct_test / total_test

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)

        logging.info(f"Epoch [{epoch+1}/{epochs}] | "
                     f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
                     f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
    
    return history

def main(args):
    """
    Main function to run SimSiam pre-training and then linear probing.
    """
    setup_logging(log_filename="simsiam.log")
    logging.info("Starting Contrastive (SimSiam) Pre-training & Linear Probing")
    logging.info(f"Arguments: {args}")

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simsiam_model = SimSiam().to(device)
    
    try:
        # --- Phase 1: SimSiam Pre-training ---
        logging.info("--- Starting Phase 1: SimSiam Pre-training ---")
        simsiam_loader = get_simsiam_loader(args.batch_size)
        
        # Optimizer and Scheduler as per SimSiam paper (scaled LR)
        init_lr = args.init_lr * args.batch_size / 256
        logging.info(f"Calculated Initial LR for SimSiam: {init_lr:.6f}")
        
        optimizer_siam = optim.SGD(simsiam_model.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
        scheduler_siam = CosineAnnealingLR(optimizer_siam, T_max=len(simsiam_loader) * args.pretrain_epochs)

        train_simsiam(
            simsiam_model, simsiam_loader, optimizer_siam, 
            scheduler_siam, device, args.pretrain_epochs
        )
        
        logging.info("--- SimSiam Pre-training Complete ---")
        
        model_path = os.path.join(output_dir, "simsiam_model.pth")
        torch.save(simsiam_model.state_dict(), model_path)
        logging.info(f"SimSiam backbone model saved to {model_path}")

        # --- Phase 2: Linear Probing ---
        logging.info("--- Starting Phase 2: Linear Probing on Labeled Data ---")
        
        # Load the pre-trained model (to ensure we're using the saved weights)
        simsiam_model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Create the classification model with the frozen backbone
        classifier_model = ClassificationModel(simsiam_model).to(device)
        
        probe_train_loader, probe_test_loader, train_size, test_size = get_baseline_loaders(
            args.batch_size, args.subset_size
        )
        
        # Optimizer *only* for the classifier head
        optimizer_probe = optim.Adam(
            filter(lambda p: p.requires_grad, classifier_model.parameters()), 
            lr=args.probe_lr
        )
        criterion_probe = nn.CrossEntropyLoss()
        
        probe_history = train_linear_probe(
            classifier_model, probe_train_loader, probe_test_loader,
            optimizer_probe, criterion_probe, device, args.probe_epochs,
            train_size, test_size
        )
        
        logging.info("--- Linear Probing Complete ---")

        # Save probe results
        model_path = os.path.join(output_dir, "simsiam_classifier_model.pth")
        torch.save(classifier_model.state_dict(), model_path)
        logging.info(f"SimSiam linear probe model saved to {model_path}")

        history_path = os.path.join(output_dir, "simsiam_ft_history.json")
        with open(history_path, 'w') as f:
            json.dump(probe_history, f)
        logging.info(f"Linear probe history saved to {history_path}")
        
        save_plot(probe_history, "SimSiam Linear Probe Performance", "simsiam_probe_metrics.png")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimSiam and Linear Probe")
    parser.add_argument("--pretrain_epochs", type=int, default=30, help="Number of SimSiam pre-training epochs")
    parser.add_argument("--probe_epochs", type=int, default=50, help="Number of linear probing epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--init_lr", type=float, default=0.05, help="Initial learning rate for SimSiam (will be scaled by batch size)")
    parser.add_argument("--probe_lr", type=float, default=1e-3, help="Learning rate for linear probe")
    parser.add_argument("--subset_size", type=int, default=5000, help="Number of labeled samples for probing")
    parser.add.argument("--no-cuda", action="store_true", default=False, help="Disables CUDA training")
    
    args = parser.parse_args()
    main(args)
