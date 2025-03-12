import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataset import CustomDataset
from dataloader import get_dataloader
from transforms import get_transform
import pandas as pd


def load_model_from_checkpoint(checkpoint_path):
    """
    Loads the trained model from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint file.
    
    Returns:
        TestModel: The loaded model set to evaluation mode.
    """
    model = TestModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


class TestModel(pl.LightningModule):
    """
    PyTorch Lightning module for testing an image classification model.
    
    Args:
        model (torch.nn.Module, optional): Pretrained model for evaluation.
    """
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def test_step(self, batch, batch_idx):
        """
        Defines the testing step, computing loss and accuracy.
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of images and labels.
            batch_idx (int): Batch index.
        
        Returns:
            torch.Tensor: Computed loss value.
        """
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
    """
    Loads a trained model, prepares the test dataset, and evaluates the model using PyTorch Lightning.
    """
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
