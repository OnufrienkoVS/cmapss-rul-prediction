import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
# import json # МБ для меты пока не знаю...
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


logger = logging.getLogger(__name__)


def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Вычисляет метрику NASA Scoring Function."""
    errors = y_pred - y_true
    scores = np.zeros_like(errors, dtype=float)
    
    late_mask = errors < 0
    scores[late_mask] = np.exp(-errors[late_mask] / 13) - 1
    
    early_mask = errors >= 0
    scores[early_mask] = np.exp(errors[early_mask] / 10) - 1
    
    return np.mean(scores)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Вычисляет все метрики для оценки модели."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nasa_score = nasa_scoring_function(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'NASA_Score': nasa_score
    }


def save_model(model, model_path, metrics=None):
    """Сохраняет модель PyTorch вместе с метаданными."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, model_path)

    logger.info(f"✅ Модель сохранена: {model_path}")


def load_model(model_path, model_class, device='cuda'):
    """
    Загружает модель PyTorch с метаданными.
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Обучает модель одну эпоху."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
    
    return total_loss / len(train_loader.dataset)


def evaluate_model_loss(model, val_loader, criterion, device):
    """Оценивает модель на валидации."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            predictions.extend(y_pred.cpu().numpy().flatten())
            targets.extend(y_batch.cpu().numpy().flatten())
    
    return total_loss / len(val_loader.dataset), np.array(predictions), np.array(targets)


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=20, wd=1e-4, device='cuda'):
    """Полный цикл обучения с early stopping."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    logger.info(f"Начало обучения на {device}")
    logger.info(f"{'Epoch':>8} | {'Train Loss':>12} | {'Val Loss':>10} | {'LR':>10}")
    logger.info("-" * 55)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate_model_loss(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"{epoch+1:>8} | {train_loss:>12.4f} | {val_loss:>10.4f} | "
                f"{optimizer.param_groups[0]['lr']:>10.2e}"
            )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping на эпохе {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    logger.info(f"Лучший Val Loss: {best_val_loss:.4f}")
    
    return model, history


def evaluate_on_test(model, test_loader, device):
    """Оценивает модель на тестовых данных."""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            
            predictions.extend(y_pred.cpu().detach().numpy().flatten())
            targets.extend(y_batch.cpu().detach().numpy().flatten())

    y_pred = np.array(predictions)
    y_test = np.array(targets)
    
    metrics = evaluate_model(y_test, y_pred)
    return metrics, y_pred