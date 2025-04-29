import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import os


def train_model(model, train_loader, val_loader, cfg, device="cuda"):
    """
    Train the model using the provided data loaders and configuration

    Args:
        model: The model to train (should be freshly initialized for each fold)
        train_loader: DataLoader for training data specific to the current fold
        val_loader: DataLoader for validation data specific to the current fold
        cfg: Configuration object containing training parameters
        device: Device to use for training

    Returns:
        history: Dictionary of training history
        val_preds: Validation predictions
        val_labels: True validation labels
    """
    print("Training a fresh model...")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_factor,
        patience=cfg.training.scheduler_patience,
        verbose=True,
    )

    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    final_val_preds = []
    final_val_labels = []

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs} [Train]"
        ):
            images = batch["image"].to(device) if model.use_images else None
            audio = batch["audio"].to(device) if model.use_audio else None

            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(x_img=images, x_audio=audio)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            train_total += batch_size
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / train_total if train_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        epoch_val_preds = []
        epoch_val_labels = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs} [Val]"
            ):
                images = batch["image"].to(device) if model.use_images else None
                audio = batch["audio"].to(device) if model.use_audio else None

                labels = batch["label"].to(device)

                outputs = model(x_img=images, x_audio=audio)
                loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                val_total += batch_size
                val_correct += (predicted == labels).sum().item()

                epoch_val_preds.extend(predicted.cpu().numpy())
                epoch_val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if cfg.wandb.enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            model_save_path = os.path.join(
                cfg.logging.output_dir, "best_contact_sound_model.pth"
            )
            torch.save(model.state_dict(), model_save_path)

            if cfg.wandb.enabled and cfg.wandb.get("upload_model", False):
                wandb.save(model_save_path)

            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")

            final_val_preds = epoch_val_preds
            final_val_labels = epoch_val_labels
        else:
            patience_counter += 1
            print(
                f"Validation accuracy did not improve. Patience: {patience_counter}/{cfg.training.early_stopping_patience}"
            )

            if patience_counter >= cfg.training.early_stopping_patience:
                print(
                    f"Early stopping triggered after epoch {epoch + 1}. Best epoch was {best_epoch + 1} with validation accuracy {best_val_acc:.4f}"
                )
                break

    print(
        f"Training complete. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}"
    )

    return history, final_val_preds, final_val_labels


def evaluate_model(model, data_loader, device="cuda"):
    """
    Evaluate the model on a given dataset.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset to evaluate on
        device: Device to run evaluation on

    Returns:
        predictions, true_labels, accuracy
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch["image"].to(device) if model.use_images else None
            audio = batch["audio"].to(device) if model.use_audio else None
            labels = batch["label"].to(device)

            outputs = model(x_img=images, x_audio=audio)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total if total > 0 else 0
    return all_preds, all_labels, accuracy

