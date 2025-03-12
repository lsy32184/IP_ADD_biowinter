import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataset import CustomDataset
from dataloader import get_dataloader
from transforms import get_transform
import pandas as pd


def load_model_from_checkpoint(checkpoint_path):
    """
    Load the trained model from a checkpoint file.
    """
    model = TestModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


class TestModel(pl.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        acc = (predicted == labels).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss


def main():
    checkpoint_path = "D:\Soyeon\Project\checkpoints\epoch=15-step=41584.ckpt"  # Change to the actual checkpoint path

    # Load model from checkpoint
    model = load_model_from_checkpoint(checkpoint_path)

    # Load test dataset
    path_metadata = r"D:\Soyeon\Project\metadata_test_R.csv"
    image_dir = r"C:\Users\BDA_INT01\Desktop\Soyeon Lee\Project\test-processed"
    image_resize = 299
    batch_size = 64
    num_workers = 0

    df = pd.read_csv(path_metadata)
    transform = get_transform(split="val", resize=image_resize)
    dataset = CustomDataset(df, image_dir, transform)
    dataloader = get_dataloader(
        dataset,
        batch_size,
        num_workers,
        shuffle=False,
        cutmix_prob=0.0,
        alpha=0.0,
        augment_prob=0.0,
    )

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer()
    trainer.test(model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
