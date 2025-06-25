import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch import nn

from utils.pannuke_dataset import PanNukeDataset
from models.attention_unet import AttentionUNet

# ---------------- Lightning Module -------------------
class LitSegModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, n_classes: int = 6):
        super().__init__()
        self.save_hyperparameters()
        self.model = AttentionUNet(n_classes=n_classes)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch['image'], batch['mask']
        logits = self(imgs)
        loss = F.cross_entropy(logits, masks)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch['image'], batch['mask']
        logits = self(imgs)
        loss = F.cross_entropy(logits, masks)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': sched,
                'monitor': 'val_loss',
            }
        }

# ---------------- Data -------------------

def get_dataloaders(batch_size=8, num_workers=4):
    ds_train = PanNukeDataset(root='Dataset', parts=['Part 1', 'Part 2', 'Part 3'], image_dirname='Images', mask_dirname='Masks', img_size=(256,256))

    # simple split (90/10)
    val_size = int(0.1 * len(ds_train))
    train_size = len(ds_train) - val_size
    ds_train, ds_val = torch.utils.data.random_split(ds_train, [train_size, val_size])

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# ---------------- Main -------------------
if __name__ == '__main__':
    pl.seed_everything(42)
    train_loader, val_loader = get_dataloaders()

    model = LitSegModel(lr=1e-3)
    summary, params, trainable = model.model.summary()
    print(summary)
    print(f"Total params: {params}, trainable: {trainable}")

    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        filename='unet-{epoch:02d}-{val_loss:.4f}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',
        devices='auto',
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_cb, lr_monitor],
    )

    trainer.fit(model, train_loader, val_loader) 