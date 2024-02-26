import torch
import pytorch_lightning as pl
from transformers import BertForSequenceClassification
from config import Config
from train_utils import load_object
from metrics import get_metrics


class BertClassifier(pl.LightningModule):  # noqa: WPS214
    def __init__(self, config: Config):
        super(BertClassifier, self).__init__()  # noqa: WPS608
        self._config = config
        self._model = BertForSequenceClassification.from_pretrained(self._config.model_path)
        self._loss_fn = torch.nn.CrossEntropyLoss()
        metrics = get_metrics(
            num_classes=self._config.num_classes,
            num_labels=self._config.num_classes,
            task='multiclass',
            average='macro',
            threshold=0.5,
        )
        self._train_metrics = metrics.clone(prefix='train_')
        self._valid_metrics = metrics.clone(prefix='val_')

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        return self._model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']
        outputs = self.forward(input_ids, attention_mask)
        loss = self._loss_fn(outputs.logits, targets)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)

        self.log('train_loss', loss)
        self._train_metrics(torch.argmax(outputs.logits, dim=1), targets)
        return loss

    def validation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']
        outputs = self.forward(input_ids, attention_mask)
        loss = self._loss_fn(outputs.logits, targets)

        self.log('val_loss', loss)
        self._valid_metrics(torch.argmax(outputs.logits, dim=1), targets)
        return loss

    def on_train_epoch_start(self):
        self._train_metrics.reset()

    def on_validation_epoch_start(self):
        self._valid_metrics.reset()

    def on_train_epoch_end(self):
        self.log_dict(self._train_metrics.compute(), on_epoch=True)

    def on_validation_epoch_end(self):
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
