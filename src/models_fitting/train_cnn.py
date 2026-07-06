import torch.nn as nn
import logging
import torch
from pathlib import Path

from models_fitting.utils.model_utils import (
    save_model,
    train_model,
    evaluate_on_test
)

from models_fitting.utils.data_utils import (
    load_data,
    create_dataloaders
)


logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN1D(nn.Module):
    def __init__(self, input_size, window_size, num_filters=64, kernel_size=3, dropout=0.3):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size + 4, padding='same')
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size + 2, padding='same')
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        self.pool3 = nn.MaxPool1d(2)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, window_size, input_size), нужно поменять местами каналы (input_size) и последовательности (window_size)
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    

def train_and_save(dataset: str, data_dir: str = '../../data/processed', model_dir: str = '../../models') -> None:
    """Обучает и сохраняет модель."""
    logger.info(f"Обучение на {dataset}")
    script_dir = Path(__file__).parent.absolute()
    
    # Загрузка данных для обучения
    data_path = (script_dir / data_dir).resolve()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path, dataset)
    
    # Обучение модели
    input_size = X_train.shape[2]
    window_size = X_train.shape[1]
    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test=y_test)

    model = CNN1D(input_size, window_size)
    model, history = train_model(model, train_loader, val_loader, device=device)
    logger.info(f"Модель обучена!")

    print()
    logger.info("ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ")

    test_metrics, _ = evaluate_on_test(model, test_loader, device)

    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Сохранение модели
    model_path = (script_dir / model_dir / f'cnn_{dataset}.pt').resolve()
    save_model(model, model_path, test_metrics)
    print()


def train_for_all_datasets(
    data_dir: str = '../../data/processed',
    model_dir: str = '../../models'
) -> None:
    """Обучает CNN модели для всех датасетов C-MAPSS."""
    logger.info("Запуск обучения CNN на всех датасетах...")

    datasets = ['FD001', 'FD002', 'FD003', 'FD004']

    for dataset in datasets:
        try:
            train_and_save(dataset, data_dir, model_dir)
        except Exception as e:
            logger.error(f"❌ {dataset} - ошибка: {e}")
            continue


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    train_for_all_datasets()