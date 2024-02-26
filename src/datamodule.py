import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import RuDataset
from config import DataConfig


class DataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self._config = config
        self._tokenizer = BertTokenizer.from_pretrained(self._config.tokenizer_path)
        self._batch_size = self._config.batch_size
        self._n_workers = self._config.n_workers

    def prepare_data(self):
        self._train_data = pd.read_csv(f'{self._config.data_path}/train.csv')
        self._valid_data = pd.read_csv(f'{self._config.data_path}/valid.csv')
        self.x_train = self.train_data['text']
        self.y_train = self.train_data['label']
        self.x_valid = self.valid_data['text']
        self.y_valid = self.valid_data['label']

    def setup(self):
        self.train_dataset = RuDataset(
            self.x_train,
            self.y_train,
            tokenizer=self._tokenizer,
            max_len=self._config.max_len,
        )
        self.val_dataset = RuDataset(
            self.x_valid,
            self.y_valid,
            tokenizer=self._tokenizer,
            max_len=self._config.max_len,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=False,
        )
