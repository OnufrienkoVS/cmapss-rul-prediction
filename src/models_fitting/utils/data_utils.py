import numpy as np
import pandas as pd
from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset


logger = logging.getLogger(__name__)


def load_data(data_dir: str, dataset_name: str):
    """Загружает последовательности для глубокого обучения."""
    data_path = Path(data_dir) / dataset_name / 'sequences'
    
    X_train = np.load(data_path / 'X_train_seq.npy')
    y_train = np.load(data_path / 'y_train_seq.npy')

    X_val = np.load(data_path / 'X_val_seq.npy')
    y_val = np.load(data_path / 'y_val_seq.npy')

    X_test = np.load(data_path / 'X_test_seq.npy')
    
    y_path = Path(data_dir) / dataset_name / 'y_test.csv'
    y_test = pd.read_csv(y_path).to_numpy(dtype='float32').reshape(-1) if y_path.exists() else None
    
    logger.info(f"Загружены данные для {dataset_name}")
    logger.info(f"  Train: X {X_train.shape}, y {y_train.shape}")
    logger.info(f"  Val:   X {X_val.shape}, y {y_val.shape}")
    if y_test is not None:
        logger.info(f"  Test:  X {X_test.shape}, y {y_test.shape}")
    else:
        logger.info(f"  Test:  X {X_test.shape}, y missing")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, batch_size=32, y_test=None):
    """Создает DataLoader для PyTorch."""
    # Преобразование в тензоры
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).view(-1, 1)

    X_test_t = torch.FloatTensor(X_test)
    
    # Dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if y_test is not None:
        y_test_t = torch.FloatTensor(y_test).view(-1, 1)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = DataLoader(TensorDataset(X_test_t), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader