import pytorch_lightning as pl
from torch.optim import Adam
import numpy as np


class Classifier(pl.LightningModule):

    def __init__(self, backbone, learning_rate: float = 0.01):
        super().__init__()
        self.save_hyperparameters('learning_rate')
        self.model = backbone
        self.forward = self.model.forward
        self.acc = pl.metrics.Accuracy()

    def step(self, batch, split):
        outputs = self(*batch)
        self.log(f'{split}_loss', outputs[0])
        return outputs

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')[0]

    def validation_step(self, batch, batch_idx):
        loss, logits = self.step(batch, 'val')
        y_hat = logits.argmax(dim=1)
        acc = self.acc(y_hat, batch[1])
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
