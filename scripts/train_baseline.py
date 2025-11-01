import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import os
import json
from tqdm import tqdm

# Adjust path to import from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Network
from src.data_loader import get_baseline_loaders
from src.utils import setup_logging, save_plot

def main(args):
    """
    Main function to train the baseline supervised model.
    """
    # Setup logging
    setup_logging(log_filename="baseline.log")
    logging.info("Starting Baseline Model Training")
    logging.info(f"Arguments: {args}")

    # Setup device
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    try:
        train_loader, test_loader, train_size, test_size = get_baseline_loaders(
            args.batch_size, args.subset_size
        )
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # Initialize model, criterion, optimizer
    model = Network(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': []
    }

    # Training loop
    try:
        logging.info("--- Starting Training Loop ---")
        for epoch in range(args.epochs):
            model.train()
            train_loss, correct_train, total_train = 0, 0, 0
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", unit="batch")
            for images, labels in train_bar:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                train_bar.set_postfix(loss=loss.item())

            # Testing phase
            model.eval()
            test_loss, correct_test, total_test = 0, 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            # Calculate and store metrics
            epoch_train_loss = train_loss / train_size
            epoch_train_acc = 100 * correct_train / total_train
            epoch_test_loss = test_loss / test_size
            epoch_test_acc = 100 * correct_test / total_test

            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['test_loss'].append(epoch_test_loss)
            history['test_acc'].append(epoch_test_acc)

            logging.info(f"Epoch [{epoch+1}/{args.epochs}] | "
                         f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
                         f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

        logging.info("--- Training Complete ---")

        # Save outputs
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save model
        model_path = os.path.join(output_dir, "baseline_model.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Baseline model saved to {model_path}")

        # Save history
        history_path = os.path.join(output_dir, "baseline_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f)
        logging.info(f"Training history saved to {history_path}")

        # Save plot
        save_plot(history, "Baseline Model Performance", "baseline_metrics.png")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline Supervised Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--subset_size", type=int, default=5000, help="Number of labeled samples to use")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA training")
    
    args = parser.parse_args()
    main(args)
