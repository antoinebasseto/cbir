"""
Create dataloaders depending on settings in config.py
"""

from config import config
from src.models.hyperparameters import params
from src.dataset.xray_dataset import ImageDataModule
from pathlib import Path


def get_datamodule():
    if config['dataset'] == 'sleep-edf-153':
        return SLEEP_EDF_DataModule(
            data_dir=Path('dataset', 'processed', config['dataset']),
            batch_size=params[config['model']]['batch_size']
            )
    else:
        raise NotImplementedError("Choose valid dataset in config.py")


if __name__ == '__main__':
    print(f"Loading dataset")
    datamodule = get_datamodule()
    datamodule.prepare_data()
    datamodule.setup()
    # test dataloader
    for batch in datamodule.train_dataloader():
        print(batch[0].shape)
        print(batch[1].shape)
        break
