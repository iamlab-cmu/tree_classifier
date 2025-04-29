import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import os


def train_model(model, train_loader, val_loader, cfg, device='cuda'):
    """
    Train the model using the provided data loaders and configuration
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        cfg: Configuration object containing training parameters
        device: Device to use for training
        
    Returns:
        history: Dictionary of training history
        val_preds: Validation predictions
        val_labels: True validation labels
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.learning_rate, 
        weight_decay=cfg.training.weight_decay
    )
    
    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=cfg.training.lr_factor, 
        patience=cfg.training.scheduler_patience, 
        verbose=True
    )
    
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Store predictions and true labels for confusion matrix
    final_val_preds = []
    final_val_labels = []
    
    for epoch in range(cfg.training.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.training.epochs} [Train]'):
            # Only pass the modalities that are enabled in the model
            images = batch['image'].to(device) if model.use_images else None
            audio = batch['audio'].to(device) if model.use_audio else None
            
            labels = batch['label'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_img=images, x_audio=audio)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size  # Use labels batch size instead of images
            _, predicted = torch.max(outputs, 1)
            train_total += batch_size
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total if train_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Store predictions and labels for this epoch
        epoch_val_preds = []
        epoch_val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{cfg.training.epochs} [Val]'):
                # Only pass the modalities that are enabled in the model
                images = batch['image'].to(device) if model.use_images else None
                audio = batch['audio'].to(device) if model.use_audio else None
                
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(x_img=images, x_audio=audio)
                loss = criterion(outputs, labels)
                
                # Statistics
                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size  # Use labels batch size instead of images
                _, predicted = torch.max(outputs, 1)
                val_total += batch_size
                val_correct += (predicted == labels).sum().item()
                
                # Store predictions and labels
                epoch_val_preds.extend(predicted.cpu().numpy())
                epoch_val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log metrics to wandb
        if cfg.wandb.enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        print(f'Epoch {epoch+1}/{cfg.training.epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            model_save_path = os.path.join(cfg.logging.output_dir, 'best_contact_sound_model.pth')
            torch.save(model.state_dict(), model_save_path)
            
            # Log best model to wandb only if upload_model is enabled
            if cfg.wandb.enabled and cfg.wandb.get('upload_model', False):
                wandb.save(model_save_path)
            
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
            
            # Update final predictions and labels
            final_val_preds = epoch_val_preds
            final_val_labels = epoch_val_labels
        else:
            patience_counter += 1
            print(f"Validation accuracy did not improve. Patience: {patience_counter}/{cfg.training.early_stopping_patience}")
            
            # Check if we should stop early
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"Early stopping triggered after epoch {epoch+1}. Best epoch was {best_epoch+1} with validation accuracy {best_val_acc:.4f}")
                break
    
    # Return both history and final validation predictions/labels for confusion matrix
    return history, final_val_preds, final_val_labels


def evaluate_model(model, data_loader, device='cuda'):
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
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Only pass the modalities that are enabled in the model
            images = batch['image'].to(device) if model.use_images else None
            audio = batch['audio'].to(device) if model.use_audio else None
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(x_img=images, x_audio=audio)
            _, predicted = torch.max(outputs, 1)
            
            # Statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0
    return all_preds, all_labels, accuracy 